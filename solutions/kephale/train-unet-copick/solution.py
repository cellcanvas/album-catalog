###album catalog: cellcanvas

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
    - git+https://github.com/kephale/morphospaces.git@copick
    - git+https://github.com/copick/copick.git
"""

def run():
    import logging
    import sys

    import pytorch_lightning as pl
    from monai.data import DataLoader
    from monai.transforms import (
        Compose, RandAffined, RandFlipd, RandRotate90d, EnsureChannelFirstd,
        ScaleIntensityd, Resized, ToTensord
    )
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from morphospaces.datasets import CopickDataset
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric

    # setup logging
    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # CLI arguments
    args = get_args()
    lr = args.lr
    logdir = args.logdir
    copick_config_path = args.copick_config_path
    train_run_names = args.train_run_names.split(",")
    val_run_names = args.val_run_names.split(",")
    tomo_type = args.tomo_type
    user_id = args.user_id
    session_id = args.session_id
    segmentation_type = args.segmentation_type
    voxel_spacing = args.voxel_spacing

    # patch parameters
    batch_size = 1
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.5

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = "./" + logdir

    # training parameters
    val_check_interval = 0.15
    accumulate_grad_batches = 4

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            EnsureChannelFirstd(keys=[image_key, labels_key]),
            ScaleIntensityd(keys=[image_key]),
            Resized(keys=[image_key, labels_key], spatial_size=patch_shape),
            RandFlipd(keys=[image_key, labels_key], prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotate90d(keys=[image_key, labels_key], prob=0.5, spatial_axes=(0, 1)),
            RandAffined(keys=[image_key, labels_key], prob=0.5, rotate_range=(1.5, 1.5, 1.5)),
            ToTensord(keys=[image_key, labels_key])
        ]
    )

    train_ds = CopickDataset.from_copick_project(
        copick_config_path=copick_config_path,
        run_names=train_run_names,
        tomo_type=tomo_type,
        user_id=user_id,
        session_id=session_id,
        segmentation_type=segmentation_type,
        voxel_spacing=voxel_spacing,
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            EnsureChannelFirstd(keys=[image_key, labels_key]),
            ScaleIntensityd(keys=[image_key]),
            Resized(keys=[image_key, labels_key], spatial_size=patch_shape),
            ToTensord(keys=[image_key, labels_key])
        ]
    )

    val_ds = CopickDataset.from_copick_project(
        copick_config_path=copick_config_path,
        run_names=val_run_names,
        tomo_type=tomo_type,
        user_id=user_id,
        session_id=session_id,
        segmentation_type=segmentation_type,
        voxel_spacing=voxel_spacing,
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="unet-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="unet-last",
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    class UNetSegmentation(pl.LightningModule):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
            self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            outputs = self.forward(images)
            val_loss = self.loss_function(outputs, labels)
            self.dice_metric(y_pred=outputs, y=labels)
            self.log("val_loss", val_loss)
            return val_loss

        def validation_epoch_end(self, outputs):
            dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
            self.log("val_dice", dice)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    net = UNetSegmentation(lr=lr)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, learning_rate_monitor],
        logger=logger,
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=100,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)


setup(
    group="kephale",
    name="train-unet-copick",
    version="0.0.2",
    title="Train 3D UNet for Segmentation with Copick Dataset",
    description="Train a 3D UNet network using the Copick dataset for segmentation.",
    solution_creators=["Kyle Harrington", "Zhuowen Zhao"],
    cite=[{"text": "Morphospaces team.", "url": "https://github.com/morphometrics/morphospaces"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "morphospaces"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "description": "Path to the Copick configuration file",
            "type": "string",
            "required": True,
        },
        {
            "name": "train_run_names",
            "description": "Names of the runs in the Copick project for training",
            "type": "string",
            "required": True,
        },
        {
            "name": "val_run_names",
            "description": "Names of the runs in the Copick project for validation",
            "type": "string",
            "required": True,
        },
        {
            "name": "tomo_type",
            "description": "Tomogram type in the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "user_id",
            "description": "User ID for the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "session_id",
            "description": "Session ID for the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "segmentation_type",
            "description": "Segmentation type in the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "voxel_spacing",
            "description": "Voxel spacing for the Copick project",
            "type": "float",
            "required": True,
        },
        {
            "name": "lr",
            "description": "Learning rate for the UNet training",
            "type": "float",
            "required": False,
            "default": 0.0001
        },
        {
            "name": "logdir",
            "description": "Output directory name in the current working directory. Default is checkpoints",
            "type": "string",
            "required": False,
            "default": "checkpoints",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
