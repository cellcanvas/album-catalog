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
    from monai.networks.nets import UNet
    from monai.transforms import ScaleIntensity, EnsureChannelFirst, ToTensor
    from monai.inferers import sliding_window_inference
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()

    copick_config_path = args.copick_config_path
    run_name = args.run_name
    tomo_type = args.tomo_type
    user_id = args.user_id
    session_id = args.session_id
    voxel_spacing = args.voxel_spacing
    checkpoint_path = args.checkpoint_path
    segmentation_name = args.segmentation_name

    patch_size = (96, 96, 96)
    overlap = 0.5

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

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    out_channels = checkpoint['state_dict']['model.model.2.1.conv.unit0.conv.weight'].shape[0]

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Inference
    run = root.get_run(run_name)
    prediction_seg = get_prediction_segmentation(run, user_id, session_id, voxel_spacing, segmentation_name)
    
    tomogram = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"]
    
    # Normalize function
    def normalize(patch):
        patch = EnsureChannelFirst()(patch)
        patch = ScaleIntensity()(patch)
        patch = ToTensor()(patch)
        return patch

    # Process tomogram in patches
    shape = tomogram.shape
    for z in range(0, shape[0], patch_size[0]):
        for y in range(0, shape[1], patch_size[1]):
            for x in range(0, shape[2], patch_size[2]):
                patch = tomogram[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                patch = normalize(patch)
                patch = patch.unsqueeze(0).to(model.device)
                with torch.no_grad():
                    output = sliding_window_inference(patch, patch_size, overlap, model)
                prediction_seg[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] = output.squeeze(0).cpu().numpy()

    print(f"Prediction complete. Segmentation saved as {segmentation_name}.")

setup(
    group="kephale",
    name="predict-unet-copick",
    version="0.0.2",
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
