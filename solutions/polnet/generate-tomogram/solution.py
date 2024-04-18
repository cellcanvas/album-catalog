###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - mrcfile
  - numpy
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
  - pip:
    - git+https://github.com/uermel/mrc2omezarr
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
    import os
    import re
    import numpy as np
    import mrcfile
    import zarr

    from mrc2omezarr.proc import convert_mrc_to_ngff

    # Fetch arguments
    args = get_args()

    # Setup paths and configurations
    ROOT_PATH = args.inputfile  # Assuming inputfile argument is used for the root path
    OUT_DIR = args.outputdirectory
    COPICK_ROOT = args.copick_root
    RUN_NAME = args.run_name
    
    PROTEINS_LIST = args.proteins_list.split(',')
    MB_PROTEINS_LIST = args.mb_proteins_list.split(',')
    MEMBRANES_LIST = args.membranes_list.split(',')

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Define tomography and feature extraction parameters
    # Note the copick part of the script assumes ntomos is always 1
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

    # Call the function to generate features
    from gui.core.all_features2 import all_features2
    all_features2(NTOMOS, VOI_SHAPE, OUT_DIR, VOI_OFFS, VOI_VSIZE, MMER_TRIES, PMER_TRIES,
                  MEMBRANES_LIST, [], PROTEINS_LIST, MB_PROTEINS_LIST, SURF_DEC,
                  TILT_ANGS, DETECTOR_SNR, MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA)

    # Copick prep
    def mrc_to_zarr(mrc_path, zarr_path):
        """Convert an MRC file to a Zarr file."""

        convert_mrc_to_ngff(mrc_path,
                            zarr_path,
                            permissive=True,)

    def setup_copick_directories(base_path, run_name, voxel_spacing):
        """Setup the directory structure for copick and return the path."""
        experiment_path = os.path.join(base_path, 'ExperimentRuns', run_name, f'VoxelSpacing{voxel_spacing:.3f}')
        segmentations_path = os.path.join(base_path, 'ExperimentRuns', run_name, 'Segmentations')
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(segmentations_path, exist_ok=True)
        return experiment_path, segmentations_path

    voxel_path, segmentations_path = setup_copick_directories(COPICK_ROOT, RUN_NAME, voxel_spacing)
    
    # Process all matching SNR tomogram files
    tomos_path = os.path.join(OUT_DIR, 'tomos')
    for filename in os.listdir(tomos_path):
        if re.match(r'tomo_rec_0_snr\d+\.\d+.mrc', filename):
            snr_value = re.findall(r'snr(\d+\.\d+)', filename)[0]
            mrc_path = os.path.join(tomos_path, filename)
            zarr_path = os.path.join(voxel_path, f'polnet_rec_0_snr{snr_value}.zarr')
            mrc_to_zarr(mrc_path, zarr_path)

            print(f"Converted {filename} to {zarr_path}")

    # Format:
    # |-- Segmentations/
    #        |-- [xx.yyy]_[user_id | tool_name]_[session_id | 0]_[name]-multilabel.zarr
    labels_name = f'{voxel_spacing:.3f}_polnet_0_all-multilabel.zarr'
            
    mrc_to_zarr(os.path.join(OUT_DIR, 'tomos', 'tomo_lbls_0.mrc'),
                os.path.join(segmentations_path, labels_name))

    print(f"Generated tomograms saved to {OUT_DIR}")
    
setup(
    group="polnet",
    name="generate-tomogram",
    version="0.0.6",
    title="Generate a tomogram with polnet",
    description="Generate tomograms with polnet, and save them in a Zarr.",
    solution_creators=["Jonathan Schwartz and Kyle Harrington"],
    cite=[{"text": "Martinez-Sanchez A.*, Jasnin M., Phelippeau H. and Lamm L. Simulating the cellular context in synthetic datasets for cryo-electron tomography, bioRxiv.", "url": "https://github.com/anmartinezs/polnet"}],
    tags=["synthetic data", "deep learning", "cryoet", "tomogram"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "inputfile", "type": "string", "required": True, "description": "Path to the root directory containing the data"},
        {"name": "outputdirectory", "type": "string", "required": True, "description": "Path for the output directory to save the generated files"},
        {"name": "proteins_list", "type": "string", "required": True, "description": "Comma-separated list of protein file paths"},
        {"name": "mb_proteins_list", "type": "string", "required": True, "description": "Comma-separated list of membrane protein file paths"},
        {"name": "membranes_list", "type": "string", "required": True, "description": "Comma-separated list of membrane file paths"},
        {"name": "copick_root", "type": "string", "required": True, "description": "Path to the root directory for copick output"},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the run for organizing outputs"}

    ],
    run=run,
    install=install,
    dependencies={
        "environment_file": env_file
    },
)
