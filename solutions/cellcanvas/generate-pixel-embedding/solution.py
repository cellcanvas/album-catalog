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
  - torch
  - monai
  - pytorch-lightning
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

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

    # Define the chunk size and overlap for sliding window inference
    input_chunk_size = image.chunks
    chunk_size = input_chunk_size if len(input_chunk_size) == 3 else input_chunk_size[1:]
    overlap = tuple(min(s // 4, 64) for s in chunk_size)  # Adjust overlap according to chunk size

    print(f"Processing image from run {run_name} with shape {image.shape} at voxel spacing {voxel_spacing}")
    print(f"Using chunk size: {chunk_size}, overlap: {overlap}")

    # Prepare output Zarr array in the tomogram store
    copick_features: TCopickFeatures = tomogram.new_features("embedding")
    out_array = zarr.open_array(store=copick_features.zarr(),
                                compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
                                dtype='float32',
                                dimension_separator='/',
                                shape=(net.embedding_dim, *image.shape),
                                chunks=(net.embedding_dim, *chunk_size))

    # Process each chunk
    for z in range(0, image.shape[0], chunk_size[0]):
        for y in range(0, image.shape[1], chunk_size[1]):
            for x in range(0, image.shape[2], chunk_size[2]):
                z_start = max(z - overlap[0], 0)
                z_end = min(z + chunk_size[0] + overlap[0], image.shape[0])
                y_start = max(y - overlap[1], 0)
                y_end = min(y + chunk_size[1] + overlap[1], image.shape[1])
                x_start = max(x - overlap[2], 0)
                x_end = min(x + chunk_size[2] + overlap[2], image.shape[2])

                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk = torch.from_numpy(np.expand_dims(chunk, axis=(0, 1)).astype(np.float32)).cuda()

                def predict_embedding(patch):
                    patch = (patch - patch.mean()) / patch.std()
                    return net(patch)

                with torch.no_grad():
                    result = sliding_window_inference(
                        inputs=chunk,
                        roi_size=chunk_size,
                        sw_batch_size=1,
                        predictor=predict_embedding,
                        overlap=overlap[0],  # Use the first element of overlap for sliding window inference
                        mode="gaussian",
                        sw_device=torch.device("cuda"),
                        device=torch.device("cuda"),
                    )

                result = result.cpu().numpy().squeeze()

                # Save the results in the appropriate chunk location
                out_array[:, z:z + chunk_size[0], y:y + chunk_size[1], x:x + chunk_size[2]] = result

    print(f"Features saved under feature type 'embedding'")

setup(
    group="cellcanvas",
    name="generate-pixel-embedding",
    version="0.1.0",
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
