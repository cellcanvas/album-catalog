###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args
import os
import subprocess

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - pip:
    - album
"""

def install():
    # TODO install    
    command = f"album install cellcanvas:generate-pixel-embedding:0.0.23"
    subprocess.run(command, shell=True, check=True)

def run():
    args = get_args()
    copick_directory = args.copick_directory
    checkpoint_path = args.checkpoint_path
    
    if not os.path.isdir(copick_directory):
        raise ValueError("The provided copick directory does not exist or is not a directory.")

    def is_hidden_directory(path):
        """
        Determine if a component of the path represents a hidden directory.
        """
        return any(part.startswith('.') for part in path.split(os.sep) if part)

    def contains_excluded_keyword(path, keywords=["painting", "prediction", "features.zarr"]):
        """
        Check if the path contains any of the excluded keywords.
        """
        return any(keyword in path for keyword in keywords)

    def is_zarr_directory(path):
        """
        Check if a directory is a Zarr dataset by looking for a '.zarray' file which is typical for Zarr datasets.
        """
        return any(fname.endswith('.zarray') for fname in os.listdir(path))

    def walk_and_process(directory, checkpoint_path):
        """
        Walk through the directory tree to find directories that are valid Zarr datasets and process each.
        """
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and modify dirs in-place to avoid traversing into hidden dirs
            if is_hidden_directory(root) or contains_excluded_keyword(root):
                dirs[:] = [d for d in dirs if not (d.startswith('.') or contains_excluded_keyword(os.path.join(root, d)))]
                continue

            # Check if the directory is a Zarr dataset within 'VoxelSpacing'
            if 'VoxelSpacing10' in root and is_zarr_directory(root):
                zarr_path = root
                path_parts = zarr_path.rstrip('/').split('/')
                if len(path_parts) > 1:
                    base_zarr_name = path_parts[-2].split('.zarr')[0]  # Gets 'wbp' from 'wbp.zarr'
                    final_segment = path_parts[-1]  # Gets '0' from the path 'wbp.zarr/0'
                    parent_dir = '/' + os.path.join(*path_parts[:-2])  # Gets directory path without the final segment
                    output_filename = f"{base_zarr_name}.{final_segment}_cellcanvas01_features.zarr"
                    output_directory = os.path.join(parent_dir, output_filename)

                    # Construct the command to call the existing solution
                    command = f"album run cellcanvas:generate-pixel-embedding:0.0.23 --checkpointpath {checkpoint_path} --inputfile {zarr_path} --outputdirectory {output_directory}"
                    print(f"Processing {zarr_path}...")
                    subprocess.run(command, shell=True, check=True)
                    print(f"Output saved to {output_directory}")

    walk_and_process(copick_directory, checkpoint_path)

setup(
    group="copick",
    name="generate-cellcanvas-features",
    version="0.0.13",
    title="Batch Process Zarr Files for Pixel Embedding",
    description="Automatically process all Zarr files within a specified directory structure using a SwinUNETR model.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas and Copick teams.", "url": "https://cellcanvas.org"}],
    tags=["batch processing", "zarr", "deep learning", "cryoet", "cellcanvas", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_directory", "type": "string", "required": True, "description": "Path to the copick directory containing the Zarr files to process."},
        {"name": "checkpoint_path", "type": "string", "required": True, "description": "Path to the checkpoint file of the trained SwinUNETR model."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },    
)
