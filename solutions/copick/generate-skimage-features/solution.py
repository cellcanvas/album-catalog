###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - mrcfile
  - h5py
  - numpy
  - pip
  - scikit-image
  - zarr
"""

def run():
    import os
    import mrcfile
    import zarr
    import numpy as np
    from skimage.feature import multiscale_basic_features
    import sys

    input_file = get_args().inputfile
    output_directory = get_args().outputdirectory
    intensity = get_args().intensity
    edges = get_args().edges
    texture = get_args().texture
    sigma_min = get_args().sigma_min
    sigma_max = get_args().sigma_max

    data_path = get_data_path()

    def predict_zarr(zarr_path, output_directory):
        os.makedirs(output_directory, exist_ok=True)

        # Load Zarr dataset
        dataset = zarr.open_array(zarr_path, mode='r')
        image = dataset[:]

        print(f"Processing image from {zarr_path} with shape {image.shape}")

        features = multiscale_basic_features(image, intensity=intensity, edges=edges, texture=texture, sigma_min=sigma_min, sigma_max=sigma_max)

        features = np.moveaxis(features, -1, 0)  # Move the features axis to the front (num_features, z, y, x)

        zarr.save_array(output_directory,
                        features,
                        chunks=(1, 256, 256, 256),
                        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE),
                        dtype=np.float32,
                        dimension_separator="/"
                        )

    def predict_mrc(input_file, output_directory):
        os.makedirs(output_directory, exist_ok=True)

        with mrcfile.open(input_file, permissive=True) as mrc:
            image = mrc.data

        print(f"Image {input_file} has shape {image.shape}")

        features = multiscale_basic_features(image, intensity=intensity, edges=edges, texture=texture, sigma_min=sigma_min, sigma_max=sigma_max)

        features = np.moveaxis(features, -1, 0)  # Move the features axis to the front (num_features, z, y, x)

        zarr.save_array(output_directory,
                        features,
                        chunks=(1, 256, 256, 256),
                        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE),
                        dtype=np.float32,
                        dimension_separator="/"
                        )

    if input_file.lower().endswith('.mrc'):
        predict_mrc(input_file, output_directory)
    else:
        predict_zarr(input_file, output_directory)

    print(f"Prediction output saved to {output_directory}")

setup(
    group="copick",
    name="generate-skimage-features",
    version="0.0.1",
    title="Generate Multiscale Basic Features with Scikit-Image",
    description="Compute multiscale basic features of a mrc or zarr tomogram and save them in a Zarr.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas + copick team.", "url": "https://cellcanvas.org"}],
    tags=["feature extraction", "image processing", "cryoet", "tomogram"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "inputfile", "type": "string", "required": True, "description": "Path to the input MRC file or zarr path containing the tomogram"},
        {"name": "outputdirectory", "type": "string", "required": True, "description": "Path for the output directory to save the Zarr file"},
        {"name": "intensity", "type": "boolean", "required": False, "default": True, "description": "Include intensity features"},
        {"name": "edges", "type": "boolean", "required": False, "default": True, "description": "Include edge features"},
        {"name": "texture", "type": "boolean", "required": False, "default": True, "description": "Include texture features"},
        {"name": "sigma_min", "type": "float", "required": False, "default": 0.5, "description": "Minimum sigma for Gaussian blurring"},
        {"name": "sigma_max", "type": "float", "required": False, "default": 16.0, "description": "Maximum sigma for Gaussian blurring"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
