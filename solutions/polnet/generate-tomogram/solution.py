###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args
import os

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - mrcfile
  - "numpy<2"
  - pip
  - zarr
  - scipy
  - jax
  - vtk
  - pandas
  - pynrrd
  - scikit-image
  - jupyter
  - wget
  - ipyfilechooser
  - jaxlib=0.4.6=cuda112*
  - cudatoolkit=11.2
  - cudnn>=8.1.0
  - pip:
    - git+https://github.com/kephale/mrc2omezarr
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def download_file(url, destination):    
    """Download a file from a URL to a destination path."""
    import requests
    
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def install():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/anmartinezs/polnet"])

def run():
    import re
    import numpy as np
    import mrcfile
    import zarr
    import json
    from shutil import move
    import shutil
    from copick.impl.filesystem import CopickRootFSSpec
    # from mrc2omezarr.proc import convert_mrc_to_ngff

    # Fetch arguments
    args = get_args()

    # Setup paths and configurations
    COPICK_CONFIG_PATH = args.copick_config_path  # Path to the Copick configuration file
    RUN_NAME = args.run_name
    PROTEINS_LIST = args.proteins_list.split(',')
    MB_PROTEINS_LIST = args.mb_proteins_list.split(',')
    MEMBRANES_LIST = args.membranes_list.split(',')

    # Load Copick configuration
    with open(COPICK_CONFIG_PATH, 'r') as f:
        copick_config = json.load(f)

    # Initialize Copick root
    root = CopickRootFSSpec.from_file(COPICK_CONFIG_PATH)

    # Ensure Copick run exists
    copick_run = root.get_run(RUN_NAME)
    if not copick_run:
        copick_run = root.new_run(RUN_NAME)

    # Define tomography and feature extraction parameters
    NTOMOS = 1
    VOI_SHAPE = (630, 630, 200)
    VOI_OFFS = ((4, 626), (4, 626), (4, 196))
    VOI_VSIZE = 10
    MMER_TRIES = 20
    PMER_TRIES = 100
    SURF_DEC = 0.9
    minAng, maxAng, angIncr = -60, 61, 3
    TILT_ANGS = range(minAng, maxAng, angIncr)
    DETECTOR_SNR = [0.1, 1.0]
    MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA = 0, 0, 0
    voxel_spacing = 10.000

    # Create a permanent directory in /tmp
    permanent_dir = "/tmp/polnet_output"
    TEM_DIR = os.path.join(permanent_dir, 'tem')
    TOMOS_DIR = os.path.join(permanent_dir, 'tomos')
    os.makedirs(TEM_DIR, exist_ok=True)
    os.makedirs(TOMOS_DIR, exist_ok=True)
    print(f"Using permanent directory at: {permanent_dir}")

    # Call the function to generate features
    from gui.core.all_features2 import all_features2
    all_features2(NTOMOS, VOI_SHAPE, permanent_dir, VOI_OFFS, VOI_VSIZE, MMER_TRIES, PMER_TRIES,
                  MEMBRANES_LIST, [], PROTEINS_LIST, MB_PROTEINS_LIST, SURF_DEC,
                  TILT_ANGS, DETECTOR_SNR, MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA)

    # Process all matching SNR tomogram files in the permanent directory
    for filename in os.listdir(TOMOS_DIR):
        if re.match(r'tomo_rec_0_snr\d+\.\d+.mrc', filename):
            snr_value = re.findall(r'snr(\d+\.\d+)', filename)[0]
            mrc_path = os.path.join(TOMOS_DIR, filename)

            # Ensure voxel spacing exists
            if voxel_spacing not in [vs.voxel_size for vs in copick_run.voxel_spacings]:
                voxel_spacing_entry = copick_run.new_voxel_spacing(voxel_size=voxel_spacing)
            else:
                voxel_spacing_entry = copick_run.get_voxel_spacing(voxel_spacing)

            # Add tomogram to Copick
            tomogram_name = f"tomogram_snr{snr_value}"
            copick_tomogram = voxel_spacing_entry.new_tomogram(tomogram_name)

            zarr_path = copick_tomogram.zarr().path

            # convert_mrc_to_ngff(mrc_path, zarr_path, permissive=True)
            # print(f"Converted {filename} to {zarr_path} and added to Copick")            

            # Optionally move files to a permanent location if needed
            print(f"Run: mrc2omezarr --mrc-path {mrc_path} --zarr-path {zarr_path} --permissive True")
            print(f"Moved {filename} to {mrc_path}")

    # Ensure segmentations are added to Copick
    def add_painting_segmentation(run, painting_segmentation_name, user_id="generatedPolnet", session_id="0"):
        segmentation_name = f'{voxel_spacing:.3f}_{painting_segmentation_name}_multilabel.zarr'

        segmentation = run.new_segmentation(
            voxel_spacing=voxel_spacing,
            name=painting_segmentation_name,
            session_id=session_id,
            is_multilabel=True,
            user_id=user_id,
        )

        # convert_mrc_to_ngff(os.path.join(permanent_dir, 'tomo_lbls_0.mrc'), segmentation.zarr(), permissive=True)
        print(f"Run: mrc2omezarr --mrc-path {os.path.join(permanent_dir, 'tomo_lbls_0.mrc')} --zarr-path {segmentation.zarr().path} --permissive True")
        print(f"Added segmentation {segmentation_name} to Copick")

    add_painting_segmentation(copick_run, "polnet_0_all")

setup(
    group="polnet",
    name="generate-tomogram",
    version="0.1.10",
    title="Generate a tomogram with polnet",
    description="Generate tomograms with polnet, and save them in a Zarr.",
    solution_creators=["Jonathan Schwartz and Kyle Harrington"],
    cite=[{"text": "Martinez-Sanchez A.*, Jasnin M., Phelippeau H. and Lamm L. Simulating the cellular context in synthetic datasets for cryo-electron tomography, bioRxiv.", "url": "https://github.com/anmartinezs/polnet"}],
    tags=["synthetic data", "deep learning", "cryoet", "tomogram"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration file"},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the run for organizing outputs"},
        {"name": "proteins_list", "type": "string", "required": True, "description": "Comma-separated list of protein file paths"},
        {"name": "mb_proteins_list", "type": "string", "required": True, "description": "Comma-separated list of membrane protein file paths"},
        {"name": "membranes_list", "type": "string", "required": True, "description": "Comma-separated list of membrane file paths"}
    ],
    run=run,
    install=install,
    dependencies={
        "environment_file": env_file
    },
)
