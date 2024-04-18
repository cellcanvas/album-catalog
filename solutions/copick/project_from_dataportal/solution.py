###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args
import subprocess

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - mrcfile
  - numpy
  - scikit-image
  - pip:
    - album
    - git+https://github.com/uermel/mrc2omezarr
    - cryoet-data-portal
"""

def install():
    command = "album install cellcanvas:generate-pixel-embedding:0.0.22"
    subprocess.run(command, shell=True, check=True)

def run():
    import os
    import mrcfile
    import numpy as np
    from skimage.transform import rescale
    import s3fs
    import zarr
    from cryoet_data_portal import Client, Run
    
    args = get_args()
    dataset_id = args.dataset_id
    num_runs = int(args.num_runs)
    region_exclude = eval(args.region_exclude)
    checkpoint_path = args.checkpoint_path
    copick_root_directory = args.copick_root_directory
    fs = s3fs.S3FileSystem(anon=True)  # Assuming the bucket is public

    def process_mrc_to_zarr(mrc_path, output_zarr_path, voxel_spacing, region_exclude):
        with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
            data = mrc.data[region_exclude[0]:region_exclude[1], region_exclude[2]:region_exclude[3], region_exclude[4]:region_exclude[5]]
            data = data.astype(np.float32)
            scaling_factor = mrc.voxel_size / voxel_spacing
            rescaled_data = rescale(data, scale=(scaling_factor, scaling_factor, scaling_factor), order=1, preserve_range=True, anti_aliasing=True)

        z = zarr.open(output_zarr_path, mode='w', shape=rescaled_data.shape, dtype=rescaled_data.dtype)
        z[:] = rescaled_data

    client = Client()
    runs = Run.find(client, [Run.dataset_id == dataset_id])
    runs = list(runs)[:num_runs]

    for run in runs:
        s3_prefix = run.s3_prefix
        tomogram_dir = fs.ls(os.path.join(s3_prefix, "Tomograms"))[0]
        voxel_dir = fs.ls(tomogram_dir)[0]
        canonical_tomogram_dir = os.path.join(voxel_dir, "CanonicalTomogram")
        mrc_file = fs.ls(canonical_tomogram_dir)[0]
        local_mrc_path = f"/tmp/{os.path.basename(mrc_file)}"
        
        # Download the MRC file
        fs.get(mrc_file, local_mrc_path)

        output_zarr_name = "wbp.zarr"
        output_features_name = "wbp_cellcanvas01_features.zarr"
        run_name = run.name
        run_directory = os.path.join(copick_root_directory, "ExperimentRuns", run_name, "VoxelSpacing10.000")
        
        # Ensure directories exist
        os.makedirs(run_directory, exist_ok=True)

        output_zarr_path = os.path.join(run_directory, output_zarr_name)
        output_features_directory = os.path.join(run_directory, output_features_name)

        voxel_spacing = 10  # Assuming rescaling to a fixed voxel spacing of 10
        process_mrc_to_zarr(local_mrc_path, output_zarr_path, voxel_spacing, region_exclude)

        # Convert MRC to OME-Zarr
        command = f"mrc2omezarr --permissive --mrc-path {local_mrc_path} --zarr-path {output_zarr_path}"
        subprocess.run(command, shell=True, check=True)

        # Generate embeddings
        command = f"album run cellcanvas:generate-pixel-embedding:0.0.22 --checkpointpath {checkpoint_path} --inputfile {output_zarr_path} --outputdirectory {output_features_directory}"
        subprocess.run(command, shell=True, check=True)

setup(
    group="copick",
    name="project_from_dataportal",
    version="0.0.2",
    title="Convert MRCs from a data portal dataset to zarr and Generate cellcanvas Pixel Embeddings",
    description="Processes MRC files to ZARR and generates embeddings for tomography data.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Cellcanvas and copick teams", "url": "https://cellcanvas.org"}],
    tags=["mrc", "zarr", "deep learning", "tomography"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "dataset_id", "type": "string", "required": True, "description": "Dataset ID to process."},
        {"name": "num_runs", "type": "string", "required": True, "description": "Number of runs to process from the dataset."},
        {"name": "region_exclude", "type": "string", "required": True, "description": "Tuple defining regions to exclude in the format (x_start, x_end, y_start, y_end, z_start, z_end)."},
        {"name": "checkpoint_path", "type": "string", "required": True, "description": "Path to the checkpoint file of the trained model."},
        {"name": "copick_root_directory", "type": "string", "required": True, "description": "Root directory of the copick project where outputs should be saved."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
