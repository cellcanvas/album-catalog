###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.9
  - cudatoolkit
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda
  - numpy
  - pip
  - pytorch-lightning
  - monai
  - qtpy
  - zarr
  - pip:
    - git+https://github.com/morphometrics/morphospaces.git
    - copick==0.5.5
"""

MODEL_URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    import requests
    
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def install():
    import os
    
    data_path = get_data_path()
    # Ensure the data path exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Download the model
    model_path = os.path.join(data_path, "data", "model_swinvit.pt")
    if not os.path.exists(model_path):
        download_file(MODEL_URL, model_path)

def run():
    import os
    import torch
    import zarr
    import numpy as np
    from monai.inferers import sliding_window_inference
    from copick.impl.filesystem import CopickRootFSSpec
   # from copick.models import TCopickFeatures
    from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR
    from numcodecs import Blosc
    from torch.cuda.amp import autocast
    import sys
    import torch

    print(f'Cuda is available {torch.cuda.is_available()}')
    # Dummy classes for Qt components
    class DummyQtComponent:
        def __init__(self, *args, **kwargs):
            pass

    class DummyQtWidgetsModule:
        QHBoxLayout = DummyQtComponent
        QPushButton = DummyQtComponent
        QWidget = DummyQtComponent

    sys.modules['qtpy.QtWidgets'] = DummyQtWidgetsModule()

    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    checkpoint_path = args.checkpointpath
    embedding_name = args.embedding_name

    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    if voxel_spacing_obj is None:
        raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

    image = zarr.open(tomogram.zarr(), mode='r')['0']

    net = PixelEmbeddingSwinUNETR.load_from_checkpoint(checkpoint_path, strict=False)
    net.cuda()
    net.eval()

    roi_size = (64, 64, 64)
    overlap = 0.5

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")
    print(f"Using ROI size: {roi_size}, overlap: {overlap}")

    torch.cuda.empty_cache()
    image = torch.from_numpy(np.expand_dims(image, axis=(0, 1)).astype(np.float32)).cuda()

    # Run a dummy forward pass to determine the embedding dimension
    with torch.no_grad():
        dummy_input = torch.zeros((1, 1, *roi_size), dtype=torch.float32).cuda()
        dummy_output = net(dummy_input)
        embedding_dim = dummy_output.shape[1]

    print(f"Determined embedding dimension: {embedding_dim}")

    copick_features = tomogram.new_features(embedding_name)
    out_array = zarr.open_array(store=copick_features.zarr(),
                                compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
                                dtype='float32',
                                dimension_separator='/',
                                shape=(embedding_dim, *image.shape[2:]),
                                chunks=(embedding_dim, *roi_size))

    def sliding_window_processor(image, roi_size, overlap, device):
        sw_batch_size = 1
        sw_device = device

        _, _, depth, height, width = image.shape

        stride = tuple(int(r * (1 - overlap)) for r in roi_size)
        num_steps = [int(np.ceil((dim - r) / stride[idx])) + 1 for idx, (dim, r) in enumerate(zip(image.shape[2:], roi_size))]

        for z in range(num_steps[0]):
            for y in range(num_steps[1]):
                for x in range(num_steps[2]):
                    start = [z * stride[0], y * stride[1], x * stride[2]]
                    end = [min(s + r, dim) for s, r, dim in zip(start, roi_size, image.shape[2:])]

                    window = image[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                    window = torch.nn.functional.pad(window, (0, roi_size[2] - window.shape[-1],
                                                              0, roi_size[1] - window.shape[-2],
                                                              0, roi_size[0] - window.shape[-3]))

                    window = window.to(sw_device)
                    window = (window - window.mean()) / window.std()

                    with torch.no_grad(), autocast():
                        result = net(window)

                    result_np = result.cpu().numpy().squeeze(axis=0)

                    actual_size = [min(r, end[idx] - start[idx]) for idx, r in enumerate(roi_size)]
                    
                    out_array[:, start[0]:start[0]+actual_size[0], 
                                 start[1]:start[1]+actual_size[1], 
                                 start[2]:start[2]+actual_size[2]] = result_np[:, :actual_size[0], :actual_size[1], :actual_size[2]]

                    torch.cuda.empty_cache()

    sliding_window_processor(image, roi_size, overlap, torch.device("cuda"))

    print(f"Embeddings saved under feature type '{embedding_name}'")


setup(
    group="cellcanvas",
    name="generate-pixel-embedding",
    version="0.1.8",
    title="Predict Tomogram Embeddings with SwinUNETR using Copick API",
    description="Apply a SwinUNETR model to a tomogram fetched using the Copick API to produce embeddings, and save them in a Zarr.",
    solution_creators=["Kyle Harrington"],
    tags=["prediction", "deep learning", "cryoet", "tomogram"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to be used."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to process."},
        {"name": "checkpointpath", "type": "string", "required": True, "description": "Path to the checkpoint file of the trained SwinUNETR model"},
        {"name": "embedding_name", "type": "string", "required": True, "description": "Name of the embedding to use as the feature name in Copick"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)