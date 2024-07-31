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
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pytorch
  - cudatoolkit=11.3
  - monai
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
    - pytorch-lightning
"""

def run():
    import torch
    import numpy as np
    import zarr
    import dask.array as da
    from monai.networks.nets import UNet
    from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord
    from monai.data import DataLoader, Dataset
    from monai.inferers import sliding_window_inference
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()

    copick_config_path = args.copick_config_path
    run_name = args.run_name
    tomo_type = args.tomo_type
    user_id = args.user_id
    session_id = args.session_id
    segmentation_type = args.segmentation_type
    voxel_spacing = args.voxel_spacing
    checkpoint_path = args.checkpoint_path
    segmentation_name = args.segmentation_name

    batch_size = args.batch_size
    patch_size = (96, 96, 96)
    overlap = 0.5

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    # Load the Copick root
    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_prediction_segmentation(run, user_id, session_id, voxel_spacing, segmentation_name):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=segmentation_name, voxel_size=voxel_spacing)
        if not run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type):
            return None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type):
                    return None
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    # Load the dataset
    def load_test_dataset(run_name, transform):
        run = root.get_run(run_name)
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr()
        return Dataset(data=[{"zarr_tomogram": tomogram}], transform=transform)

    transforms = Compose([
        LoadImaged(keys=[image_key]),
        AddChanneld(keys=[image_key]),
        ScaleIntensityd(keys=[image_key]),
        ToTensord(keys=[image_key]),
    ])

    test_ds = load_test_dataset(run_name, transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    out_channels = checkpoint['state_dict']['model.out.conv.weight'].shape[0]

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Inference
    run = root.get_run(run_name)
    prediction_seg = get_prediction_segmentation(run, user_id, session_id, voxel_spacing, segmentation_name)

    for batch in test_loader:
        inputs = batch[image_key].to(model.device)
        with torch.no_grad():
            outputs = sliding_window_inference(inputs, patch_size, overlap, model)

        # Save the outputs into copick
        z = 0
        for chunk in outputs:
            prediction_seg[z:z+chunk.shape[0], :, :] = chunk.cpu().numpy()
            z += chunk.shape[0]

    print(f"Prediction complete. Segmentation saved as {segmentation_name}.")

setup(
    group="kephale",
    name="predict-unet-copick",
    version="0.0.1",
    title="Generate Segmentation Masks using UNet Checkpoint",
    description="Generate segmentation masks using a trained UNet checkpoint on the Copick dataset.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Cellcanvas team.", "url": "https://cellcanvas.org"}],
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
            "name": "run_name",
            "description": "Name of the run in the Copick project for testing",
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
            "name": "checkpoint_path",
            "description": "Path to the trained UNet checkpoint",
            "type": "string",
            "required": True,
        },
        {
            "name": "segmentation_name",
            "description": "Name of the output segmentation",
            "type": "string",
            "required": True,
        },
        {
            "name": "batch_size",
            "description": "Batch size for inference",
            "type": "integer",
            "required": False,
            "default": 1,
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
