###album catalog: cellcanvas

from io import StringIO
from album.runner.api import setup

env_file = StringIO(
    """name: morphospaces
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit=11.3
  - dask
  - einops
  - h5py
  - magicgui
  - monai
  - numpy
  - pytorch-lightning
  - qtpy
  - rich
  - scikit-image
  - scipy
  - tensorboard
  - torch
  - zarr
  - "git+https://github.com/uermel/copick.git"
  - "git+https://github.com/morphometrics/morphospaces.git"
"""
)

def run():
    from album.runner.api import get_args
    import argparse
    import logging
    import sys

    import pytorch_lightning as pl
    from monai.data import DataLoader, ConcatDataset
    from monai.transforms import Compose, RandAffined, RandFlipd, RandRotate90d
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from morphospaces.datasets.zarr import LazyTiledZarrDataset
    from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR
    from morphospaces.transforms.image import ExpandDimsd, StandardizeImage
    from morphospaces.transforms.label import LabelsAsFloat32
    from copick.impl.filesystem import CopickRootFSSpec
    import zarr

    # Get CLI arguments
    args = get_args()
    lr = args.lr
    logdir_path = args.logdir_path
    batch_size = args.batch_size
    patch_threshold = args.patch_threshold
    loss_temperature = args.loss_temperature
    pretrained_weights_path = args.pretrained_weights_path
    max_epochs = args.max_epochs
    copick_config_path = args.copick_config_path
    train_run_names = args.train_run_names
    val_run_names = args.val_run_names
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type

    # setup logging
    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # patch parameters
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = f"{logdir_path}/checkpoints_swin_{learning_rate_string}_memory_20240319"

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.15
    lr_reduction_patience = 25
    lr_scheduler_step = 1500
    accumulate_grad_batches = 4
    memory_banks: bool = False
    n_pixel_embeddings_per_class: int = 1000
    n_pixel_embeddings_to_update: int = 10
    n_label_embeddings_per_class: int = 50
    n_memory_warmup: int = 1000

    pl.seed_everything(42, workers=True)

    # Load Copick configuration
    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_datasets(run_names, stage, transform):
        datasets = []
        unique_label_values = set()
        
        if run_names:
            runs = [root.get_run(run_name) for run_name in run_names.split(',')]
            for run in runs:
                if run is None:
                    raise ValueError(f"Run with name '{run.meta.name}' not found.")
        else:
            runs = root.runs

        for run in runs:
            voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
            if voxel_spacing_obj is None:
                raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run.meta.name}'.")

            # Get tomogram
            tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
            if tomogram is None:
                raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}' in run '{run.meta.name}'.")

            # Open highest resolution
            image = zarr.open(tomogram.zarr(), mode='r')['0']

            dataset = LazyTiledZarrDataset(
                file_path=image,
                stage=stage,
                transform=transform,
                patch_shape=patch_shape,
                stride_shape=patch_stride,
                patch_filter_ignore_index=(0,),
                patch_filter_key='label',
                patch_threshold=patch_threshold,
                patch_slack_acceptance=0,
                store_unique_label_values=True,
            )

            datasets.append(dataset)
            unique_label_values.update(dataset.unique_label_values)

        return ConcatDataset(datasets), unique_label_values

    train_transform = Compose(
        [
            LabelsAsFloat32(keys='label'),
            StandardizeImage(keys='raw'),
            ExpandDimsd(
                keys=[
                    'raw',
                    'label',
                ]
            ),
            RandFlipd(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.2,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.2,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.2,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.25,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.25,
                spatial_axes=(0, 2),
            ),
            RandRotate90d(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.25,
                spatial_axes=(1, 2),
            ),
            RandAffined(
                keys=[
                    'raw',
                    'label',
                ],
                prob=0.5,
                mode="nearest",
                rotate_range=(1.5, 1.5, 1.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
        ]
    )

    train_ds, unique_train_label_values = get_datasets(train_run_names, 'train', train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # get the unique train label values
    print(f"Unique train label values: {unique_train_label_values}")

    val_transform = Compose(
        [
            LabelsAsFloat32(keys='label'),
            StandardizeImage(keys='raw'),
            ExpandDimsd(
                keys=[
                    'raw',
                    'label',
                ]
            ),
        ]
    )

    val_ds, unique_val_label_values = get_datasets(val_run_names, 'val', val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Unique val label values: {unique_val_label_values}")

    # get all unique label values
    unique_label_values = set(unique_train_label_values).union(set(unique_val_label_values))
    print(f"Unique label values: {unique_label_values}")

    # make the checkpoint callback
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="pe-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="pe-last",
    )

    # learning rate monitor
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    # make the model
    net = PixelEmbeddingSwinUNETR(
        pretrained_weights_path=pretrained_weights_path,
        image_key='raw',
        labels_key='label',
        in_channels=1,
        n_embedding_dims=48,
        lr_scheduler_step=lr_scheduler_step,
        lr_reduction_patience=lr_reduction_patience,
        learning_rate=lr,
        loss_temperature=loss_temperature,
        n_samples_per_class=n_samples_per_class,
        label_values=unique_label_values,
        memory_banks=memory_banks,
        n_pixel_embeddings_per_class=n_pixel_embeddings_per_class,
        n_pixel_embeddings_to_update=n_pixel_embeddings_to_update,
        n_label_embeddings_per_class=n_label_embeddings_per_class,
        n_memory_warmup=n_memory_warmup,
    )

    # logger
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback,
            learning_rate_monitor,
        ],
        logger=logger,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        check_val_every_n_epoch=6,
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

setup(
    group="morphospaces",
    name="train_swin_unetr_pixel_embedding",
    version="0.0.1",
    title="Train SwinUnetr Pixel Embedding Network",
    description="Train the SwinUnetr pixel embedding network using the provided script and dataset.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Morphospaces team.", "url": "https://github.com/morphometrics/morphospaces"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "morphospaces"],
    license="MIT",
    covers=[
        {
            "description": "Training SwinUnetr Pixel Embedding Network",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    args=[
        {"name": "lr", "type": "float", "description": "Learning rate."},
        {"name": "logdir_path", "type": "string", "description": "Path to save logs and checkpoints."},
        {"name": "batch_size", "type": "integer", "description": "Batch size for training."},
        {"name": "patch_threshold", "type": "float", "description": "Patch threshold."},
        {"name": "loss_temperature", "type": "float", "description": "Loss temperature."},
        {"name": "pretrained_weights_path", "type": "string", "description": "Path to pretrained weights."},
        {"name": "max_epochs", "type": "integer", "description": "Maximum number of epochs for training."},
        {"name": "copick_config_path", "type": "string", "description": "Path to the Copick configuration JSON file."},
        {"name": "train_run_names", "type": "string", "description": "Comma-separated list of Copick run names for training data."},
        {"name": "val_run_names", "type": "string", "description": "Comma-separated list of Copick run names for validation data."},
        {"name": "voxel_spacing", "type": "float", "description": "Voxel spacing to be used."},
        {"name": "tomo_type", "type": "string", "description": "Type of tomogram to process."}
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
