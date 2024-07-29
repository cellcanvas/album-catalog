###album catalog: cellcanvas

from album.runner.api import setup, get_args
import os

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - h5py
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import h5py
    import numpy as np
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()
    copick_config_path = args.copick_config_path
    run_names = args.run_names.split(',')
    output_hdf_path = args.output_hdf_path
    voxel_spacing = args.voxel_spacing
    tomo_type = args.tomo_type
    user_id_filter = args.user_id
    session_id_filter = args.session_id

    # Load the Copick root from the configuration file
    root = CopickRootFSSpec.from_file(copick_config_path)

    # Create HDF5 file
    with h5py.File(output_hdf_path, 'w') as hdf:
        for run_name in run_names:
            run_name = run_name.strip()
            # Open the run
            run = root.get_run(run_name)
            if not run:
                raise ValueError(f"Run with name '{run_name}' not found.")

            # Get tomogram
            tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
            if tomogram:
                tomo_data = zarr.open(tomogram.zarr(), mode='r')['0'][:]
                hdf.create_dataset(f'{run_name}/tomograms/{tomo_type}', data=tomo_data, compression="gzip")

            # Export points
            for obj in root.config.pickable_objects:
                for pick_set in run.get_picks(obj.name):
                    if pick_set and pick_set.points:
                        if (user_id_filter and pick_set.user_id != user_id_filter) or (session_id_filter and pick_set.session_id != session_id_filter):
                            continue
                        points = np.array([[p.location.z, p.location.y, p.location.x] for p in pick_set.points])
                        hdf.create_dataset(f'{run_name}/picks/{obj.name}/{pick_set.user_id}/{pick_set.session_id}', data=points, compression="gzip")
    
    print(f"Exported tomogram and picks to {output_hdf_path}")

setup(
    group="copick",
    name="export-to-hdf",
    version="0.0.3",
    title="Export Copick Runs to HDF5",
    description="A solution that exports multiple Copick runs' tomograms and picks into an HDF5 file.",
    solution_creators=["Kyle Harrington"],
    tags=["data export", "zarr", "hdf5", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_names", "type": "string", "required": True, "description": "Comma-separated list of Copick run names to process."},
        {"name": "output_hdf_path", "type": "string", "required": True, "description": "Path to the output HDF5 file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale tomogram and pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to export (e.g., denoised)."},
        {"name": "user_id", "type": "string", "required": False, "description": "Filter picks by user ID."},
        {"name": "session_id", "type": "string", "required": False, "description": "Filter picks by session ID."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
