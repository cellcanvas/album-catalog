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
        Determine if the path includes a hidden directory.
        """
        return any(part.startswith('.') for part in path.split(os.sep))
    
    def is_valid_zarr(zarr_path):
        """
        Check if the Zarr file is valid for processing based on the specified conditions.
        Exclude files that are hidden, system files or contain specific keywords.
        """
        base_name = os.path.basename(zarr_path)
        invalid_keywords = ["painting", "prediction", "features.zarr"]
        if base_name.startswith('.') or any(keyword in base_name for keyword in invalid_keywords):
            return False
        return base_name.endswith('.zarr')

    def walk_and_process(directory, checkpoint_path):
        """
        Walk through the directory tree to find valid Zarr files and process each using the existing album solution.
        Only processes Zarr files within directories that match 'VoxelSpacing*' and skips files based on specific keywords.
        """
        for root, dirs, files in os.walk(directory):
            if 'VoxelSpacing' in root and not is_hidden_directory(root):
                for file in files:
                    if file.endswith('.zarr') and is_valid_zarr(os.path.join(root, file)) and not is_hidden_directory(os.path.join(root, file)):
                        zarr_path = os.path.join(root, file)
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
    version="0.0.4",
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
