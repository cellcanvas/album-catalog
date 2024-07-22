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
  - "numpy<2"
  - scipy
  - scikit-image
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import numpy as np
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint
    import zarr

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = args.voxel_spacing
    session_id = args.session_id
    user_id = args.user_id
    grid_spacing_factor = float(args.grid_spacing_factor)
    run_name = args.run_name if 'run_name' in args else None
    tomo_type = args.tomo_type

    # Load Copick configuration
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    def create_grid_of_picks(run, spacing_factor):
        for obj in root.pickable_objects:
            obj_name = obj.name
            radius = obj.radius
            grid_spacing = radius * spacing_factor
            
            # Get the tomogram
            voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
            if voxel_spacing_obj is None:
                raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run.name}'.")

            tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
            if tomogram is None:
                raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

            # Open the highest resolution
            image = zarr.open(tomogram.zarr(), mode='r')['0']

            # Create a grid of points
            points = []
            for z in np.arange(0, image.shape[0], grid_spacing):
                for y in np.arange(0, image.shape[1], grid_spacing):
                    for x in np.arange(0, image.shape[2], grid_spacing):
                        points.append(CopickPoint(location={'x': x, 'y': y, 'z': z}))

            # Save the picks
            pick_set = run.new_picks(obj_name, session_id, user_id)
            pick_set.points = points
            pick_set.store()
            print(f"Saved {len(points)} grid points for object {obj_name}.")

    if run_name:
        run = root.get_run(run_name)
        create_grid_of_picks(run, grid_spacing_factor)
    else:
        for run in root.runs:
            create_grid_of_picks(run, grid_spacing_factor)

setup(
    group="copick",
    name="grid-picks",
    version="0.0.1",
    title="Grid Picks from Tomogram",
    description="A solution that places a grid of picks based on the radius of each pickable object, parameterized by a multiple of the particle radius, using tomogram shape.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "grid", "picks", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "grid_spacing_factor", "type": "float", "required": True, "description": "Factor to multiply the particle radius by to determine grid spacing."},
        {"name": "run_name", "type": "string", "required": False, "description": "Name of the Copick run to process. If not specified, process all runs."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to process."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing to be used."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
