###album catalog: mycatalog

from album.runner.api import get_args, setup

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python==3.9
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit
  - pytorch-cuda
  - dask
  - einops
  - h5py
  - magicgui
  - monai
  - numpy<2
  - pytorch-lightning
  - qtpy
  - rich
  - scikit-image
  - scipy
  - tensorboard
  - mrcfile
  - pip:
    - git+https://github.com/zhuowenzhao/morphospaces.git@czii_cryoet
    - PyQt5
"""


def run():
    import logging
    import sys

    import pytorch_lightning as pl
    from monai.data import DataLoader
    from monai.transforms import Compose, RandAffined, RandFlipd, RandRotate90d
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from morphospaces.datasets import MrcDataset
    from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR
    from morphospaces.transforms.image import ExpandDimsd, StandardizeImage
    from morphospaces.transforms.label import LabelsAsFloat32

    # setup logging
    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


    # CLI arguments
    args = get_args()
    lr = args.lr
    logdir = args.logdir
    pretrained_weights_path = args.pretrained_weights_path
    train_tomogram_path = args.train_tomogram_path
    train_mask_path = args.train_mask_path
    val_tomogram_path = args.val_tomogram_path
    val_mask_path = args.val_mask_path
    # lr = 0.0008

    # patch parameters
    batch_size = 1
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.5

    loss_temperature = 0.1

    image_key = "mrc_tomogram"
    labels_key = "mrc_mask"
    train_data_pattern =  {image_key: train_tomogram_path+'/*.mrc',
                        labels_key: train_mask_path+'/*.mrc'}
    val_data_pattern =  {image_key: val_tomogram_path+'/*.mrc',
                        labels_key: val_mask_path+'/*.mrc'}


    print(f'train_data_pattern {train_data_pattern}')
    
    learning_rate_string = str(lr).replace(".", "_")
    #logdir_path = f"./checkpoints_swin_{learning_rate_string}_memory_20240513"
    logdir_path = "./"+logdir

    # pretrained weights
    pretrained_weights_path = pretrained_weights_path

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.15
    lr_reduction_patience = 25
    lr_scheduler_step = 1500
    # lr_scheduler_step = 1000
    accumulate_grad_batches = 4
    memory_banks: bool = True #False
    n_pixel_embeddings_per_class: int = 1000
    n_pixel_embeddings_to_update: int = 10
    n_label_embeddings_per_class: int = 50
    n_memory_warmup: int = 1000

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 2),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(1, 2),
            ),
            RandAffined(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.5,
                mode="nearest",
                rotate_range=(1.5, 1.5, 1.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
        ]
    )
    (
        train_ds,
        unique_train_label_values,
    ) = MrcDataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
        store_unique_label_values=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # print(f'\ntrain dataloader shape {next(iter(train_loader))[image_key].shape}\n')
    # get the unique label values
    #print(f"Unique train label values: shape {len(unique_train_label_values)}\n{unique_train_label_values}")

    val_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            ),
        ]
    )
    val_ds, unique_val_label_values = MrcDataset.from_glob_pattern(
        glob_pattern=val_data_pattern,
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
        store_unique_label_values=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    #print(f"Unique val label values: shape {len(unique_val_label_values)}\n{unique_val_label_values}")

    # get all unique label values
    unique_label_values = set(unique_train_label_values).union(
        set(unique_val_label_values)
    )
    #print(f"Unique label values: shape {len(unique_label_values)}\n{unique_label_values}")

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
        image_key=image_key,
        labels_key=labels_key,
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
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


setup(
    group="morphospaces",
    name="train_swin_unetr_pixel_embedding_mrc",
    version="0.0.1",
    title="Train SwinUnetr Pixel Embedding Network with Mrc Dtataloader",
    description="Train the SwinUnetr pixel embedding network using the provided script and dataset.",
    solution_creators=["Kevin Yamauchi", "Kyle Harrington", "Zhuowen Zhao"],
    cite=[{"text": "Morphospaces team.", "url": "https://github.com/morphometrics/morphospaces"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "morphospaces"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "pretrained_weights_path",
            "description": "Pretrained weights path",
            "type": "string",
            "required": True,
        },
        {
            "name": "train_tomogram_path",
            "description": "tomogram mrc files directory for training dataset",
            "type": "string",
            "required": True,
        },
        {
            "name": "train_mask_path",
            "description": "mask mrc files directory for taining dataset",
            "type": "string",
            "required": True,
        },
        {
            "name": "val_tomogram_path",
            "description": "tomogram mrc files directory for validation dataset",
            "type": "string",
            "required": True,
        },
        {
            "name": "val_mask_path",
            "description": "mask mrc files directory for validation dataset",
            "type": "string",
            "required": True,
        },
        {
            "name": "lr",
            "description": "Learning reate for the supervised contrastive learning",
            "type": "float",
            "required": False,
            "default": 0.0001
        },
        {
            "name": "logdir",
            "description": "output dir name in curr working dir. Default is checkpoints",
            "type": "string",
            "required": False,
            "default": "checkpoints",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
