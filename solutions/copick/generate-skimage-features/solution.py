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

    # Open highest resolution
    image = zarr.open(tomogram.zarr(), mode='r')['0']

    # Determine chunk size from input Zarr
    input_chunk_size = image.chunks
    chunk_size = input_chunk_size if len(input_chunk_size) == 3 else input_chunk_size[1:]
    
    # Determine overlap based on sigma_max
    overlap = int(3 * sigma_max)  # Using 3 * sigma_max to ensure enough overlap for Gaussian blur

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")
    print(f"Using chunk size: {chunk_size}, overlap: {overlap}")

    # Determine number of features by running on a small test array
    test_chunk = np.zeros((10, 10, 10), dtype=image.dtype)
    test_features = multiscale_basic_features(
        test_chunk,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )
    num_features = test_features.shape[-1]

    # Prepare output Zarr array directly in the tomogram store
    print(f"Creating new feature store with {num_features} features...")
    copick_features: TCopickFeatures = tomogram.new_features(
        feature_type,
        shape=(num_features, *image.shape),
        chunks=(num_features, *chunk_size),
        compressor=dict(id='blosc', cname='zstd', clevel=3, shuffle=2),
        dtype='float32',
        dimension_separator='/'
    )
    out_array = copick_features.zarr()

    # Process each chunk
    for z in range(0, image.shape[0], chunk_size[0]):
        for y in range(0, image.shape[1], chunk_size[1]):
            for x in range(0, image.shape[2], chunk_size[2]):
                z_start = max(z - overlap, 0)
                z_end = min(z + chunk_size[0] + overlap, image.shape[0])
                y_start = max(y - overlap, 0)
                y_end = min(y + chunk_size[1] + overlap, image.shape[1])
                x_start = max(x - overlap, 0)
                x_end = min(x + chunk_size[2] + overlap, image.shape[2])
                
                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk_features = multiscale_basic_features(
                    chunk,
                    intensity=intensity,
                    edges=edges,
                    texture=texture,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max
                )

                # Adjust indices for overlap
                z_slice = slice(z_start + overlap - z, z_end - overlap)
                y_slice = slice(y_start + overlap - y, y_end - overlap)
                x_slice = slice(x_start + overlap - x, x_end - overlap)
                
                # Ensure contiguous array before writing
                contiguous_chunk = np.ascontiguousarray(chunk_features[z_slice, y_slice, x_slice].transpose(3, 0, 1, 2))

                out_array[:, z:z + chunk_size[0], y:y + chunk_size[1], x:x + chunk_size[2]] = contiguous_chunk

    print(f"Features saved under feature type '{feature_type}'")

setup(
    group="copick",
    name="generate-skimage-features",
    version="0.1.8",
    title="Generate Multiscale Basic Features with Scikit-Image using Copick API (Chunked)",
    description="Compute multiscale basic features of a tomogram from a Copick run in chunks and save them using Copick's API.",
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
        {"name": "sigma_max", "type": "float", "required": False, "default": 16.0, "description": "Maximum sigma for Gaussian blurring"}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
