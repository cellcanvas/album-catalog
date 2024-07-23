###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - pytorch
  - monai
  - nibabel
  - scikit-image
  - ignite
  - tensorboard
  - torchvision
  - einops
  - transformers
  - pip:
    - album
    - mlflow
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import glob
    import numpy as np
    import zarr
    from monai.config import print_config
    from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader, Dataset
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        ScaleIntensityRanged,
        CropForegroundd,
        Orientationd,
        Spacingd,
        RandCropByPosNegLabeld,
        EnsureTyped,
        Activations,
        AsDiscrete,
    )
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.inferers import sliding_window_inference
    from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
    from ignite.handlers import ModelCheckpoint, EarlyStopping
    from monai.handlers import (
        MeanDice,
        MLFlowHandler,
        StatsHandler,
        TensorBoardImageHandler,
        TensorBoardStatsHandler,
    )
    
    from ignite.metrics import Loss
    import torch
    import mlflow
    import mlflow.pytorch
    from copick.impl.filesystem import CopickRootFSSpec

    print_config()

    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names.split(',')
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    seg_type = args.seg_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    experiment_name = args.experiment_name

    # Set up mlflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

    # Log parameters
    mlflow.log_params({
        "copick_config_path": copick_config_path,
        "run_names": run_names,
        "voxel_spacing": voxel_spacing,
        "tomo_type": tomo_type,
        "seg_type": seg_type,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    # Load the Copick root from the configuration file
    root = CopickRootFSSpec.from_file(copick_config_path)

    data_dicts = []

    for run_name in run_names:
        run = root.get_run(run_name)
        if not run:
            raise ValueError(f"Run with name '{run_name}' not found.")

        voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
        if not voxel_spacing_obj:
            raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

        tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
        segmentation = run.get_segmentations(name=seg_type, voxel_size=voxel_spacing)[0]

        if not tomogram:
            raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")
        if not segmentation:
            raise ValueError(f"Segmentation type '{seg_type}' not found for voxel spacing '{voxel_spacing}'.")

        data_dicts.append({"image": tomogram.zarr(), "label": segmentation.zarr()})

    # Define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # Split data into training and validation sets
    val_split = int(0.2 * len(data_dicts))
    train_files, val_files = data_dicts[val_split:], data_dicts[:val_split]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    # Initialize UNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Loss, optimizer, and metrics
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dice_metric = MeanDice(include_background=False)

    # Create trainer
    trainer = create_supervised_trainer(model, optimizer, loss_function, device, False)

    # Setup event handlers for checkpointing and logging
    log_dir = "./logs"
    checkpoint_handler = ModelCheckpoint(log_dir, "net", n_saved=10, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"model": model, "optimizer": optimizer},
    )

    # StatsHandler prints loss at every iteration
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    # MLFlowHandler plots loss at every iteration on MLFlow web UI
    mlflow_dir = os.path.join(log_dir, "mlruns")
    train_mlflow_handler = MLFlowHandler(tracking_uri=mlflow_dir, output_transform=lambda x: x)
    train_mlflow_handler.attach(trainer)

    # Optional section for model validation during training
    validation_every_n_epochs = 1
    # Set parameters for validation
    metric_name = "Mean_Dice"
    val_metrics = {metric_name: MeanDice()}
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    evaluator = create_supervised_evaluator(
        model,
        metrics=val_metrics,
        device=device,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)],
        ),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
    def run_validation(engine):
        evaluator.run(val_loader)

    # Add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_stats_handler.attach(evaluator)

    # Add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_tensorboard_stats_handler.attach(evaluator)

    # Add handler to record metrics to MLFlow at every validation epoch
    val_mlflow_handler = MLFlowHandler(
        tracking_uri=mlflow_dir,
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_mlflow_handler.attach(evaluator)

    # Add handler to draw the first image and the corresponding
    # label and model output in the last batch
    val_tensorboard_image_handler = TensorBoardImageHandler(
        log_dir=log_dir,
        batch_transform=lambda batch: (batch[0], batch[1]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda x: trainer.state.epoch,
    )
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=val_tensorboard_image_handler,
    )

    # Training loop
    trainer.run(train_loader, max_epochs=num_epochs)

    # Save the model checkpoint
    mlflow.pytorch.log_model(model, "model")

    print("Training completed.")
    mlflow.end_run()

setup(
    group="kephale",
    name="train-unet",
    version="0.0.6",
    title="Train UNet Model using MONAI with Multiple Runs and MLflow",
    description="Train a UNet model to predict segmentation masks using MONAI from multiple runs with MLflow tracking.",
    solution_creators=["Kyle Harrington"],
    tags=["segmentation", "deep learning", "monai", "unet", "mlflow"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_names", "type": "string", "required": True, "description": "Comma-separated list of Copick run names to process."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to be used."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to process."},
        {"name": "seg_type", "type": "string", "required": True, "description": "Type of segmentation labels to use."},
        {"name": "num_epochs", "type": "integer", "required": False, "default": 50, "description": "Number of training epochs."},
        {"name": "batch_size", "type": "integer", "required": False, "default": 4, "description": "Batch size for training."},
        {"name": "learning_rate", "type": "float", "required": False, "default": 1e-4, "description": "Learning rate for the optimizer."},
        {"name": "experiment_name", "type": "string", "required": True, "description": "Name of the MLflow experiment."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
