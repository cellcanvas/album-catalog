###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - torch
  - monai
  - nibabel
  - scikit-image
  - pip:
    - album
    - mlflow
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import numpy as np
    import zarr
    import torch
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose,
        AddChannel,
        ScaleIntensity,
        EnsureType,
        ToTensor,
        RandRotate90,
    )
    from monai.data import Dataset, DataLoader, CacheDataset
    from monai.losses import DiceLoss
    from monai.optimizers import Novograd
    from monai.metrics import DiceMetric
    from copick.impl.filesystem import CopickRootFSSpec
    import mlflow
    import mlflow.pytorch

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

    images = []
    labels = []

    for run_name in run_names:
        run = root.get_run(run_name)
        if not run:
            raise ValueError(f"Run with name '{run_name}' not found.")

        voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
        if not voxel_spacing_obj:
            raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

        # Get tomogram and segmentation
        tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
        segmentation = voxel_spacing_obj.get_segmentation(seg_type)

        if not tomogram:
            raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")
        if not segmentation:
            raise ValueError(f"Segmentation type '{seg_type}' not found for voxel spacing '{voxel_spacing}'.")

        images.append(zarr.open(tomogram.zarr(), mode='r')['0'])
        labels.append(zarr.open(segmentation.zarr(), mode='r')['data'])

    # Transformations
    train_transforms = Compose([
        AddChannel(),
        ScaleIntensity(),
        RandRotate90(prob=0.5, spatial_axes=[0, 1]),
        EnsureType(),
        ToTensor()
    ])
    val_transforms = Compose([
        AddChannel(),
        ScaleIntensity(),
        EnsureType(),
        ToTensor()
    ])

    # Combine images and labels from all runs
    all_images = np.concatenate(images, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    # Create Dataset and DataLoader
    dataset = CacheDataset(
        data=[{"image": img, "label": lbl} for img, lbl in zip(all_images, all_labels)],
        transform=train_transforms
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize UNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Loss, optimizer, and metrics
    loss_function = DiceLoss(sigmoid=True)
    optimizer = Novograd(model.parameters(), lr=learning_rate)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        # Log the loss
        mlflow.log_metric("loss", epoch_loss, step=epoch)

    # Save the model checkpoint
    mlflow.pytorch.log_model(model, "model")

    print("Training completed.")
    mlflow.end_run()

setup(
    group="kephale",
    name="train-unet",
    version="0.0.1",
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
