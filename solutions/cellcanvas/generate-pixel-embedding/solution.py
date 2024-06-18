###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - cudatoolkit=11.8
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - numpy
  - pip
  - pytorch-lightning
  - monai
  - qtpy
  - zarr
  - pip:
    - git+https://github.com/morphometrics/morphospaces.git
    - git+https://github.com/uermel/copick.git
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
    from copick.models import TCopickFeatures
    from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR
    from numcodecs import Blosc
    import sys

    # Define a dummy class for the expected Qt components
    class DummyQtComponent:
        def __init__(self, *args, **kwargs):
            pass

    # Define a dummy module with stubs for the Qt components
    class DummyQtWidgetsModule:
        QHBoxLayout = DummyQtComponent
        QPushButton = DummyQtComponent
        QWidget = DummyQtComponent

    # Replace 'qtpy.QtWidgets' in sys.modules with the dummy module
    sys.modules['qtpy.QtWidgets'] = DummyQtWidgetsModule()

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    checkpoint_path = args.checkpointpath

    # Load Copick configuration
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    # Get run and voxel spacing
    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    if voxel_spacing_obj is None:
        raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

    # Get tomogram
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

    # Open highest resolution
    image = zarr.open(tomogram.zarr(), mode='r')['0']

    # Load the model checkpoint
    net = PixelEmbeddingSwinUNETR.load_from_checkpoint(checkpoint_path)

    # Define the ROI size and overlap for sliding window inference
    roi_size = (64, 64, 64)  # Adjust based on typical image sizes
    overlap = 0.5  # Set overlap to ensure good stitching

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")
    print(f"Using ROI size: {roi_size}, overlap: {overlap}")

    # Prepare output Zarr array directly in the tomogram store
    copick_features: TCopickFeatures = tomogram.new_features("embedding")
    out_array = zarr.open_array(store=copick_features.zarr(),
                                compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
                                dtype='float32',
                                dimension_separator='/',
                                shape=(net.embedding_dim, *image.shape),
                                chunks=(net.embedding_dim, *roi_size))

    # Perform inference
    image = torch.from_numpy(np.expand_dims(image, axis=(0, 1)).astype(np.float32))
    image = image.cuda()

    def predict_embedding(patch):
        patch = (patch - patch.mean()) / patch.std()
        return net(patch)

    with torch.no_grad():
        result = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=predict_embedding,
            overlap=overlap,
            mode="gaussian",
            sw_device=torch.device("cuda"),
            device=torch.device("cuda"),
        )

    result = result.cpu().numpy().squeeze()

    # Save the results
    out_array[:] = result

    print(f"Embeddings saved under feature type 'embedding'")

setup(
    group="cellcanvas",
    name="generate-pixel-embedding",
    version="0.1.1",
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
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
