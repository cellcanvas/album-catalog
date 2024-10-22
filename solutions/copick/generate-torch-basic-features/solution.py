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
    - copick
"""

def run():
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import torchvision.transforms.v2 as transforms
    import numpy as np
    import copick
    import zarr
    from numcodecs import Blosc
    import os

    def gaussian_kernel_3d(kernel_size, sigma):
        """Create a 3D Gaussian kernel."""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy, zz = torch.meshgrid([ax, ax, ax])
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel

    def apply_gaussian_blur_3d(tensor, sigma):
        """Apply 3D Gaussian blur using a Conv3d layer."""
        kernel_size = max(3, int(6 * sigma + 1))
        kernel = gaussian_kernel_3d(kernel_size, sigma).to(tensor.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        conv = nn.Conv3d(1, 1, kernel_size, padding=0, bias=False)  # No padding to avoid artificial shift
        conv.weight.data = kernel
        conv.weight.requires_grad = False  # No need to train this filter
        return conv(F.pad(tensor, (kernel_size // 2,) * 6, mode='reflect'))  # Reflect padding to handle edges

    def compute_features(chunk_tensor, sigma_min, sigma_max, num_sigma, intensity, edges, texture):
        features = []

        # Add channel dimension to 3D chunk
        chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(0)

        # Intensity (raw intensity values)
        if intensity:
            features.append(chunk_tensor)  # No modification needed for intensity

        if edges:
            # Sobel kernels for 3D edge detection (no padding, handle edge effects with reflection)
            sobel_kernel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                        [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                                        [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(chunk_tensor.device)

            sobel_kernel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                        [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                                        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(chunk_tensor.device)

            sobel_kernel_z = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(chunk_tensor.device)

            grad_x = F.conv3d(F.pad(chunk_tensor, (1,) * 6, mode='reflect'), sobel_kernel_x)
            grad_y = F.conv3d(F.pad(chunk_tensor, (1,) * 6, mode='reflect'), sobel_kernel_y)
            grad_z = F.conv3d(F.pad(chunk_tensor, (1,) * 6, mode='reflect'), sobel_kernel_z)

            edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            features.append(edge_magnitude)

        # Texture (using 3D Gaussian blur at different scales)
        if texture:
            sigmas = torch.logspace(np.log10(sigma_min), np.log10(sigma_max), steps=num_sigma)
            for sigma in sigmas:
                blurred = apply_gaussian_blur_3d(chunk_tensor, sigma)
                features.append(blurred)

        # Gradient Magnitude (if the tensor is large enough)
        if chunk_tensor.shape[2] > 1 and chunk_tensor.shape[3] > 1 and chunk_tensor.shape[4] > 1:
            gradient_magnitude = torch.sqrt(torch.sum(torch.stack(torch.gradient(chunk_tensor.squeeze(0).squeeze(0))), dim=0))
            features.append(gradient_magnitude.unsqueeze(0).unsqueeze(0))
        else:
            # If tensor is too small, append a placeholder (zeros)
            print(f"Skipping gradient magnitude computation due to small tensor size: {chunk_tensor.shape}")
            features.append(torch.zeros_like(chunk_tensor))

        # Ensure that all feature maps are trimmed to the original chunk size
        min_z = min(f.shape[2] for f in features)
        min_y = min(f.shape[3] for f in features)
        min_x = min(f.shape[4] for f in features)

        features_trimmed = [f[:, :, :min_z, :min_y, :min_x] for f in features]

        # Concatenate all features to maintain consistent shape
        return torch.cat(features_trimmed, dim=1).squeeze(0)  # Concatenating along the channel dimension

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
    num_sigma = args.num_sigma if args.num_sigma else 5

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
    copick_features = tomogram.get_features(feature_type)
    if not copick_features:
        copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    # Create the Zarr array for features
    num_features = 1  # for gradient magnitude
    if intensity:
        num_features += 1
    if edges:
        num_features += 1  # Assuming one edge magnitude feature
    if texture:
        num_features += num_sigma  # Adjusting based on the number of texture scales

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

                print(f"Processing {z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}")

                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]

                # Convert to PyTorch tensor and move to device
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)

                # Compute features using PyTorch
                chunk_features = compute_features(chunk_tensor, sigma_min, sigma_max, num_sigma, intensity, edges, texture)

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
    version="0.0.6",
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
        {"name": "sigma_max", "type": "float", "required": False, "default": 16.0, "description": "Maximum sigma for Gaussian blurring"},
        {"name": "num_sigma", "type": "integer", "required": False, "default": 5, "description": "Number of sigma values between sigma_min and sigma_max for texture features."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
