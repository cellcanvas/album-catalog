###album catalog: cellcanvas

from album.runner.api import get_args, setup

def run():
    import torch
    import pytorch_lightning as pl
    from monai.networks.nets import UNet
    from monai.data import DataLoader
    from monai.losses import DiceCELoss, DiceLoss, FocalLoss, TverskyLoss
    from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
    from copick_torch import data, transforms, training, log_setup
    import mlflow
    import numpy as np
    import torch.nn as nn

    args = get_args()

    copick_config_path = args.copick_config_path
    train_run_names = args.train_run_names
    val_run_names = args.val_run_names
    tomo_type = args.tomo_type
    user_id = args.user_id
    session_id = args.session_id
    segmentation_type = args.segmentation_type
    voxel_spacing = args.voxel_spacing
    lr = args.lr
    logdir = args.logdir
    experiment_name = args.experiment_name
    num_classes = args.num_classes

    batch_size = args.batch_size
    max_epochs = args.max_epochs
    num_res_units = args.num_res_units

    logdir_path = "./" + logdir

    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.01

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    log = log_setup.setup_logging()

    train_transform = transforms.get_train_transform(image_key, labels_key)
    val_transform = transforms.get_val_transform(image_key, labels_key)

    train_ds, unique_train_label_values = data.load_dataset(
        copick_config_path, train_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, train_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    val_ds, unique_val_label_values = data.load_dataset(
        copick_config_path, val_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, val_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    unique_label_values = set(unique_train_label_values).union(set(unique_val_label_values))
    num_classes = num_classes + 1  # Adding 1 for background class

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])

    class CombinedLoss(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
            super(CombinedLoss, self).__init__()
            self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
            self.tversky_loss = TverskyLoss(to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)
            self.focal_loss = FocalLoss(to_onehot_y=True, gamma=gamma)

        def forward(self, outputs, labels):
            dice = self.dice_loss(outputs, labels)
            tversky = self.tversky_loss(outputs, labels)
            focal = self.focal_loss(outputs, labels)
            return dice + tversky + focal
    
    class UNetSegmentation(pl.LightningModule):
        def __init__(self, lr, num_classes, num_res_units):
            super().__init__()
            self.lr = lr
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(1, 1, 1, 1),
                num_res_units=num_res_units,
            )
            # self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
            self.loss_function = CombinedLoss()
            self.val_outputs = []

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()

            # Sample regions containing only 0 values and call the whole patch background
            unlabeled_mask = (labels == 0)
            random_background = unlabeled_mask
            labels[random_background] = num_classes - 1  # Assign background class

            # Log unique values in labels to debug
            unique_labels = torch.unique(labels)
            print(f"Training step unique labels: {unique_labels}")

            if torch.any(labels >= num_classes) or torch.any(labels < 0):
                raise ValueError(f"Invalid label values detected in training batch: {unique_labels}")

            labels = labels.unsqueeze(1)  # Ensure labels have the correct shape

            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()

            unlabeled_mask = (labels == 0)
            random_background = unlabeled_mask
            labels[random_background] = num_classes - 1  # Assign background class

            unique_labels = torch.unique(labels)
            print(f"Validation step unique labels: {unique_labels}")

            if torch.any(labels >= num_classes) or torch.any(labels < 0):
                raise ValueError(f"Invalid label values detected in validation batch: {unique_labels}")

            labels = labels.unsqueeze(1)  # Ensure labels have the correct shape

            outputs = self.forward(images)
            outputs = post_pred(outputs)

            if batch_idx == 0:
                text_log = {
                    "Debug/Images Shape": str(images.shape),
                    "Debug/Labels Shape": str(labels.shape),
                    "Debug/Outputs Shape": str(outputs.shape),
                    "Debug/Labels Unique Values": str(torch.unique(labels).tolist()),
                    "Debug/Outputs Unique Values": str(torch.unique(outputs).tolist())
                }
                mlflow.log_dict(text_log, "validation_debug_info.json")

            try:
                val_loss = self.loss_function(outputs, labels)
                self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                mlflow.log_metric("val_loss", val_loss.item(), step=self.global_step)

                # Log individual loss components
                dice_loss = self.loss_function.dice_loss(outputs, labels).item()
                tversky_loss = self.loss_function.tversky_loss(outputs, labels).item()
                focal_loss = self.loss_function.focal_loss(outputs, labels).item()

                self.log("val_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_tversky_loss", tversky_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

                mlflow.log_metric("val_dice_loss", dice_loss, step=self.global_step)
                mlflow.log_metric("val_tversky_loss", tversky_loss, step=self.global_step)
                mlflow.log_metric("val_focal_loss", focal_loss, step=self.global_step)
            except RuntimeError as e:
                print(f"Validation loss computation failed: {e}")
                print(f"Output shape: {outputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Output unique values: {torch.unique(outputs)}")
                print(f"Label unique values: {torch.unique(labels)}")
                raise e

            self.val_outputs.append(val_loss)
            return val_loss

        def on_validation_epoch_end(self):
            self.val_outputs.clear()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    net = UNetSegmentation(lr=lr, num_classes=num_classes, num_res_units=num_res_units)

    training.train_model(net, train_loader, val_loader, lr, logdir_path, 100, 0.15, 4, max_epochs=max_epochs, model_name="unet", experiment_name=experiment_name)


setup(
    group="kephale",
    name="train-unet-copick",
    version="0.0.34",
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
        {
            "name": "experiment_name",
            "description": "mlflow experiment name. Default is unet_experiment",
            "type": "string",
            "required": False,
            "default": "unet_experiment",
        },
        {
            "name": "batch_size",
            "description": "Batch size for training",
            "type": "integer",
            "required": False,
            "default": 1,
        },
        {
            "name": "max_epochs",
            "description": "Maximum number of epochs for training",
            "type": "integer",
            "required": False,
            "default": 10000,
        },
        {
            "name": "num_res_units",
            "description": "Number of residual units in the UNet model",
            "type": "integer",
            "required": False,
            "default": 2,
        },
        {
            "name": "num_classes",
            "description": "Number output classes",
            "type": "integer",
            "required": False,
            "default": 2,
        },        
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "copick-monai",
            "version": "0.0.2"
        }
    }
)
