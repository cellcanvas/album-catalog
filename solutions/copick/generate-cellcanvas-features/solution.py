###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args
import os
import subprocess

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - pip:
    - album
"""

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
            if 'VoxelSpacing' in root and is_zarr_directory(root):
                zarr_path = root
                output_directory = f"{zarr_path}_cellcanvas01_features.zarr"

                # Construct the command to call the existing solution
                command = f"album run cellcanvas:generate-pixel-embedding:0.0.21 --checkpointpath {checkpoint_path} --inputfile {zarr_path} --outputdirectory {output_directory}"
                print(f"Processing {zarr_path}...")
                subprocess.run(command, shell=True, check=True)
                print(f"Output saved to {output_directory}")

    walk_and_process(copick_directory, checkpoint_path)


setup(
    group="copick",
    name="generate-cellcanvas-features",
    version="0.0.6",
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
