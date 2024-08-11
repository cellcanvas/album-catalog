###album catalog: cellcanvas

from album.runner.api import get_args, setup

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
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
    - git+https://github.com/kephale/copick_torch.git
    - git+https://github.com/kephale/pixeltapestery.git
"""

def run():
    import logging
    import sys

    import torch
    import pytorch_lightning as pl
    from monai.data import DataLoader
    from monai.transforms import Compose, RandAffined, RandFlipd, RandRotate90d
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from morphospaces.datasets import CopickDataset

    from pixeltapestery.models import BinaryEmbeddingSwinUNETR
    from copick_torch import data, transforms, logging

    # setup logging
    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # CLI arguments
    args = get_args()
    lr = args.lr
    logdir = args.logdir
    pretrained_weights_path = args.pretrained_weights_path
    copick_config_path = args.copick_config_path
    train_run_names = args.train_run_names
    train_run_names = train_run_names.split(",")
    val_run_names = args.val_run_names
    val_run_names = val_run_names.split(",")
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

    loss_temperature = 0.1

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = "./" + logdir

    # pretrained weights
    pretrained_weights_path = pretrained_weights_path

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.15
    lr_reduction_patience = 25
    lr_scheduler_step = 1500
    accumulate_grad_batches = 4
    memory_banks: bool = True
    n_pixel_embeddings_per_class: int = 1000
    n_pixel_embeddings_to_update: int = 10
    n_label_embeddings_per_class: int = 50
    n_memory_warmup: int = 1000

    pl.seed_everything(42, workers=True)

    train_transform = transforms.get_train_transform(image_key, labels_key)
    val_transform = transforms.get_val_transform(image_key, labels_key)

    train_ds, unique_train_label_values = data.load_dataset(
        copick_config_path, train_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, train_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    val_ds, unique_val_label_values = data.load_dataset(
        copick_config_path, val_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, val_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    unique_label_values = set(unique_train_label_values).union(set(unique_val_label_values))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="binary-pe-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="binary-pe-last",
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    net = BinaryEmbeddingSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=len(unique_label_values),
        feature_size=48,
        spatial_dims=3,
        pretrained_weights_path=pretrained_weights_path,
    )

    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    class BinarySwinUNETRSegmentation(pl.LightningModule):
        def __init__(self, lr, model):
            super().__init__()
            self.lr = lr
            self.model = model
            self.loss_function = torch.nn.BCEWithLogitsLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.float()
            sigmoid_pixel_embedding, logits = self.forward(images)
            loss = self.loss_function(logits, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.float()
            sigmoid_pixel_embedding, logits = self.forward(images)
            val_loss = self.loss_function(logits, labels)
            self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return val_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    model = BinarySwinUNETRSegmentation(lr=lr, model=net)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, learning_rate_monitor],
        logger=logger,
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


setup(
    group="pixeltapestry",
    name="train_binary_swin_unetr_embedding_copick",
    version="0.0.2",
    title="Train Binary SwinUNETR Pixel Embedding Network with Copick Dataset",
    description="Train the Binary SwinUNETR pixel embedding network using the Copick dataset.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Pixeltapestry team.", "url": "https://github.com/kephale/pixeltapestry"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "pixeltapestery"],
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
            "description": "Learning rate for the supervised contrastive learning",
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
