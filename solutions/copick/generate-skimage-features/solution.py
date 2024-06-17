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
  - "numpy<2"
  - scipy
  - scikit-image
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import numpy as np
    from skimage.feature import multiscale_basic_features
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import TCopickFeatures
    import zarr

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    feature_type = args.feature_type
    intensity = args.intensity
    edges = args.edges
    texture = args.texture
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max

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

    image = zarr.open(tomogram.zarr(), mode='r')[:]

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")

    # Compute multiscale basic features
    features = multiscale_basic_features(
        image,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )

    # Move the features axis to the front (num_features, z, y, x)
    features = np.moveaxis(features, -1, 0)

    print("Saving features using Copick...")
    copick_features: TCopickFeatures = tomogram.new_features(
        feature_type,
        data=features,
        chunks=(1, 256, 256, 256),
        compressor=dict(id='blosc', cname='zstd', clevel=3, shuffle=2),
        dtype='float32',
        dimension_separator='/'
    )
    copick_features.store()
    print(f"Features saved under feature type '{feature_type}'")

setup(
    group="copick",
    name="generate-skimage-features",
    version="0.1.1",
    title="Generate Multiscale Basic Features with Scikit-Image using Copick API",
    description="Compute multiscale basic features of a tomogram from a Copick run and save them using Copick's API.",
    solution_creators=["Kyle Harrington"],
    tags=["feature extraction", "image processing", "cryoet", "tomogram"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to be used."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to process."},
        {"name": "feature_type", "type": "string", "required": True, "description": "Name for the feature type to be saved."},
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
