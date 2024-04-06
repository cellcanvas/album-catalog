###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - cudatoolkit=11.8
  - cuda-version=11.8
  - pytorch[version='==2.2.1']
  - torchvision
  - mrcfile
  - h5py
  - numpy
  - pip
  - pytorch-lightning
  - monai
  - qtpy
  - zarr
  - pip:
    - git+https://github.com/morphometrics/morphospaces.git
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
    import mrcfile
    import zarr
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
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
    
    from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR

    data_path = get_data_path()
    model_path = os.path.join(data_path, "model_swinvit.pt")

    def predict_mrc(input_file, output_directory, checkpoint_path, roi_size=(64, 64, 64), overlap=0.5, stitching_mode="gaussian"):
        os.makedirs(output_directory, exist_ok=True)

        with mrcfile.open(input_file, permissive=True) as mrc:
            image = mrc.data

        print(f"Image {input_file} has shape {image.shape}")
            
        image = torch.from_numpy(np.expand_dims(image, axis=(0, 1)).astype(np.float32))

        net = PixelEmbeddingSwinUNETR.load_from_checkpoint(checkpoint_path)
        
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
                mode=stitching_mode,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        input_file_base = os.path.basename(input_file)
        output_path = os.path.join(output_directory, input_file_base + '.zarr')
        
        root = zarr.open(output_path, mode='w')
        root.create_dataset(
            "raw",
            data=np.squeeze(image.cpu().numpy()),
            chunks=(64, 64, 64),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            dtype=np.float32
        )
        root.create_dataset(
            "embedding",
            data=np.squeeze(result.cpu().numpy()),
            chunks=(1, 64, 64, 64),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            dtype=np.float32
        )

    input_file = get_args().inputfile
    output_directory = get_args().outputdirectory
    checkpoint_path = get_args().checkpointpath

    predict_mrc(input_file, output_directory, checkpoint_path)

    print(f"Prediction output saved to {output_directory}")

setup(
    group="cellcanvas",
    name="generate-pixel-embedding",
    version="0.0.11",
    title="Predict Tomogram Segmentations with SwinUNETR",
    description="Apply a SwinUNETR model to MRC tomograms to produce embeddings, and save them in a Zarr.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas team.", "url": "https://cellcanvas.org"}],
    tags=["prediction", "deep learning", "cryoet", "tomogram"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "inputfile", "type": "string", "required": True, "description": "Path to the input MRC file containing the tomogram"},
        {"name": "outputdirectory", "type": "string", "required": True, "description": "Path for the output directory to save the H5 file"},
        {"name": "checkpointpath", "type": "string", "required": True, "description": "Path to the checkpoint file of the trained SwinUNETR model"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
