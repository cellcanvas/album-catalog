###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args
import os

def run():
    import re
    import numpy as np
    import zarr
    import json
    from shutil import move
    import shutil
    from copick.impl.filesystem import CopickRootFSSpec
    import mrcfile

    def convert_mrc_to_zarr(mrc_path, zarr_path, group_name, dtype):
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.astype(dtype)
            
            # Normalize the data
            mean = np.mean(data)
            std = np.std(data)
            data = (data - mean) / std
            
            print(f"Data shape: {data.shape}")
            print(f"Data dtype: {data.dtype}")
            print(f"Data sample (first 10 elements): {data.flatten()[:10]}")

        zarr_group = zarr.open_group(zarr_path, mode='w')
        zarr_dataset = zarr_group.create_dataset(group_name, data=data, chunks=True, compression='gzip')
        print(f"Zarr dataset shape: {zarr_dataset.shape}")
        print(f"Zarr dataset dtype: {zarr_dataset.dtype}")
        print(f"Zarr dataset sample (first 10 elements): {zarr_dataset[:10]}")

        print(f"Converted {mrc_path} to {zarr_path}/{group_name}")

    # Fetch arguments
    args = get_args()

    def split_args(arg):
        return arg.split(',') if arg else []

    # Setup paths and configurations
    COPICK_CONFIG_PATH = args.copick_config_path  # Path to the Copick configuration file
    RUN_NAME = args.run_name
    PROTEINS_LIST = split_args(args.proteins_list)
    MB_PROTEINS_LIST = split_args(args.mb_proteins_list)
    MEMBRANES_LIST = split_args(args.membranes_list)
    USER_ID = args.user_id
    SESSION_ID = args.session_id
    SEGMENTATION_NAME = args.segmentation_name
    TOMO_TYPE = args.tomo_type
    RETURN_PROTEIN_LABELS_ONLY = args.return_protein_labels_only.lower() == 'true'

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
            tomogram_name = f"{TOMO_TYPE}"
            copick_tomogram = voxel_spacing_entry.new_tomogram(tomogram_name)

            zarr_path = copick_tomogram.zarr().path

            convert_mrc_to_zarr(mrc_path, zarr_path, "0", dtype='float32')
            print(f"Converted {filename} to {zarr_path}/0 and added to Copick")

    def add_painting_segmentation(run, painting_segmentation_name, user_id, session_id, return_protein_labels_only):
        segmentation = run.new_segmentation(
            voxel_spacing,
            name=painting_segmentation_name,
            session_id=session_id,
            is_multilabel=True,
            user_id=user_id,
        )

        mrc_path = os.path.join(permanent_dir, 'tomos/tomo_lbls_0.mrc')
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.astype('int32')
            if return_protein_labels_only:
                num_proteins = len(PROTEINS_LIST)
                max_protein_label = data.max() - num_proteins
                data = np.where(data > max_protein_label, data - max_protein_label, 0)
                data = np.where(data > 0, data, 0)

        zarr_group = zarr.open_group(segmentation.zarr().path, mode='w')
        zarr_dataset = zarr_group.create_dataset("data", data=data, chunks=True, compression='gzip')

        print(f"Added segmentation {painting_segmentation_name} to Copick")
        print(f"Segmentation shape: {data.shape}")
        print(f"Segmentation dtype: {data.dtype}")
        print(f"Segmentation sample (first 10 elements): {data.flatten()[:10]}")

    add_painting_segmentation(copick_run, SEGMENTATION_NAME, USER_ID, SESSION_ID, RETURN_PROTEIN_LABELS_ONLY)

setup(
    group="polnet",
    name="generate-tomogram",
    version="0.1.21",
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
        {"name": "membranes_list", "type": "string", "required": True, "description": "Comma-separated list of membrane file paths"},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for Copick"},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for Copick"},
        {"name": "segmentation_name", "type": "string", "required": True, "description": "Name for the segmentation in Copick"},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram for naming in Copick"},
        {"name": "return_protein_labels_only", "type": "string", "required": False, 'default': True, "description": "Return only the labels for proteins (true/false)"}
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "polnet",
            "version": "0.0.2"
        }
    }
)
