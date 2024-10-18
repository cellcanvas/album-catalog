###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - zarr
  - "numpy<2"
  - scipy
  - joblib
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2 as transforms
    import numpy as np
    import copick
    import zarr
    from numcodecs import Blosc
    import os

    def compute_features(chunk_tensor, sigma_min, sigma_max, intensity, edges, texture):
        features = []

        # Intensity (simply the raw intensity values)
        if intensity:
            features.append(chunk_tensor.unsqueeze(0))

        # Edges (using Sobel filters)
        if edges:
            sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=chunk_tensor.device).unsqueeze(0).unsqueeze(0)
            sobel_kernel_y = sobel_kernel_x.transpose(2, 3)
            sobel_kernel_z = sobel_kernel_x.transpose(1, 3)

            grad_x = F.conv3d(chunk_tensor.unsqueeze(0), sobel_kernel_x, padding=1)
            grad_y = F.conv3d(chunk_tensor.unsqueeze(0), sobel_kernel_y, padding=1)
            grad_z = F.conv3d(chunk_tensor.unsqueeze(0), sobel_kernel_z, padding=1)

            edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            features.append(edge_magnitude)

        # Texture (using Gaussian blurs at different scales)
        if texture:
            for sigma in torch.linspace(sigma_min, sigma_max, steps=5):
                blur_transform = transforms.GaussianBlur(kernel_size=int(6*sigma+1), sigma=sigma.item())
                blurred = blur_transform(chunk_tensor.unsqueeze(0))
                features.append(blurred)

        # Additional features: Laplacian of Gaussian (LoG) for edge detection
        laplacian = F.conv3d(chunk_tensor.unsqueeze(0), torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=chunk_tensor.device), padding=1)
        features.append(laplacian)

        # Additional features: Gradient Magnitude
        gradient_magnitude = torch.sqrt(torch.sum(torch.stack(torch.gradient(chunk_tensor)), dim=0))
        features.append(gradient_magnitude.unsqueeze(0))

        return torch.cat(features, dim=0)

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
    root = copick.from_file(copick_config_path)
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

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare output Zarr array directly in the tomogram store
    print(f"Creating new feature store...")
    copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    # Create the Zarr array for features
    num_features = 0  # Placeholder for number of features
    out_array = zarr.create(
        shape=(num_features, *image.shape),
        chunks=(num_features, *chunk_size),
        dtype='float32',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
        store=feature_store,
        overwrite=True
    )

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

                # Convert to PyTorch tensor and move to device
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)

                # Compute features using PyTorch
                chunk_features = compute_features(chunk_tensor, sigma_min, sigma_max, intensity, edges, texture)

                # Adjust indices for overlap
                z_slice = slice(overlap if z_start > 0 else 0, None if z_end == image.shape[0] else -overlap)
                y_slice = slice(overlap if y_start > 0 else 0, None if y_end == image.shape[1] else -overlap)
                x_slice = slice(overlap if x_start > 0 else 0, None if x_end == image.shape[2] else -overlap)

                # Store features
                out_array[:, z:z + chunk_size[0], y:y + chunk_size[1], x:x + chunk_size[2]] = chunk_features[:, z_slice, y_slice, x_slice].cpu().numpy()

    print(f"Features saved under feature type '{feature_type}'")

setup(
    group="copick",
    name="generate-torch-basic-features",
    version="0.0.2",
    title="Generate Multiscale Basic Features with Torch using Copick API (Chunked, Corrected)",
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
