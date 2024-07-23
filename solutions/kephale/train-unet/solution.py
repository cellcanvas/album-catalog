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
  - numpy
  - pytorch
  - monai
  - scikit-image
  - ignite
  - nibabel
  - pynrrd
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
    import numpy as np
    import torch
    from monai.config import print_config
    from monai.data import DataLoader, CacheDataset, MetaTensor
    from monai.transforms import (
        Compose, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, Spacingd, EnsureTyped, Activations, AsDiscrete, Resized, RandFlipd, RandRotate90d, RandZoomd
    )
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.handlers import MeanDice, TensorBoardImageHandler
    from ignite.engine import Events, Engine, create_supervised_evaluator
    from ignite.handlers import ModelCheckpoint
    from ignite.metrics import Loss
    import mlflow
    import mlflow.pytorch
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec

    print_config()

    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names.split(',')
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    seg_type = args.seg_type
    num_epochs = args.num_epochs if args.num_epochs else 100
    batch_size = args.batch_size if args.batch_size else 4
    learning_rate = args.learning_rate if args.learning_rate else 1e-4
    experiment_name = args.experiment_name

    # Set up mlflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(vars(args))

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
        
        if not tomogram or not segmentation:
            raise ValueError(f"Missing data for run '{run_name}', voxel spacing '{voxel_spacing}', tomogram type '{tomo_type}', or segmentation type '{seg_type}'.")

        # Convert Zarr data to NumPy arrays and add a channel dimension
        tomogram_store = zarr.open(tomogram.zarr(), mode='r')
        segmentation_store = zarr.open(segmentation.zarr(), mode='r')
        tomogram_data = np.expand_dims(np.array(tomogram_store["0"]), axis=0)
        segmentation_data = np.expand_dims(np.array(segmentation_store["data"]), axis=0)

        # Wrap in MetaTensor
        tomogram_data = MetaTensor(tomogram_data, meta={"original_affine": np.eye(4), "original_channel_dim": 0})
        segmentation_data = MetaTensor(segmentation_data, meta={"original_affine": np.eye(4), "original_channel_dim": 0})

        data_dicts.append({"image": tomogram_data, "label": segmentation_data})

    # Define transforms for image and segmentation
    transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),  # Resize all images and labels to the same size
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1)
    ])

    val_split = max(int(0.2 * len(data_dicts)), 1)
    train_files, val_files = data_dicts[val_split:], data_dicts[:val_split]

    # Ensure non-empty datasets
    if not train_files:
        raise ValueError("Training dataset is empty. Please provide non-empty data.")
    if not val_files:
        raise ValueError("Validation dataset is empty. Please provide non-empty data.")

    print(f"Number of training samples: {len(train_files)}")
    print(f"Number of validation samples: {len(val_files)}")

    train_ds = CacheDataset(data=train_files, transform=transforms)
    val_ds = CacheDataset(data=val_files, transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dice_metric = MeanDice(include_background=False)

    def prepare_batch(batch, device=None, non_blocking=False):
        x = batch["image"]
        y = batch["label"]
        return (
            x.to(device=device, non_blocking=non_blocking),
            y.to(device=device, non_blocking=non_blocking),
        )

    def supervised_update_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device, non_blocking=True)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}] Loss: {loss.item()}")
        return loss.item()  # Ensure the loss is returned as a float
    
    trainer = Engine(supervised_update_function)
    checkpoint_handler = ModelCheckpoint("./logs", "net", n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model, "optimizer": optimizer})

    val_metrics = {"Mean_Dice": dice_metric}
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = AsDiscrete(threshold=0.5)

    evaluator = create_supervised_evaluator(
        model,
        metrics={"Loss": Loss(loss_function), "Mean_Dice": dice_metric},
        device=device,
        non_blocking=True,
        prepare_batch=prepare_batch,
    )
    
    tensorboard_image_handler = TensorBoardImageHandler(
        log_dir="./logs",
        batch_transform=lambda batch: (batch["image"], batch["label"]),
        output_transform=lambda output: output,
        global_iter_transform=lambda x: trainer.state.epoch
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, tensorboard_image_handler)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def run_validation(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        mean_dice = metrics["Mean_Dice"]
        print(f"Validation Results - Epoch: {engine.state.epoch} Mean Dice: {mean_dice:.4f}")
        mlflow.log_metric("mean_dice", mean_dice, step=engine.state.epoch)

    trainer.run(train_loader, max_epochs=num_epochs)

    # Log model with mlflow
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

setup(
    group="kephale",
    name="train-unet",
    version="0.0.10",
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
        {"name": "num_epochs", "type": "integer", "required": False, "default": 100, "description": "Number of training epochs."},
        {"name": "batch_size", "type": "integer", "required": False, "default": 4, "description": "Batch size for training."},
        {"name": "learning_rate", "type": "float", "required": False, "default": 1e-4, "description": "Learning rate for the optimizer."},
        {"name": "experiment_name", "type": "string", "required": True, "description": "Name of the MLflow experiment."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
