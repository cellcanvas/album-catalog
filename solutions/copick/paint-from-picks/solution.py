###album catalog: cellcanvas

from album.runner.api import setup, get_args
import os

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - pip:
    - album
    - "copick[all]"
    - "git+https://github.com/copick/copick-utils.git"
"""

def run():
    import numpy as np
    import zarr
    import copick

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = args.voxel_spacing
    ball_radius_factor = args.ball_radius_factor
    run_name = args.run_name
    allowlist_user_ids = args.allowlist_user_ids
    tomo_type = args.tomo_type

    if allowlist_user_ids:
        allowlist_user_ids = allowlist_user_ids.split(',')
    else:
        allowlist_user_ids = []

    # Load the Copick root from the configuration file
    root = copick.from_file(copick_config_path)

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "paintingsegmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        if not tomogram:
            print(f"Cannot find tomogram with voxel spacing =  {voxel_spacing} and tomo type = {tomo_type} in run {run}")
            print(f"Voxel spacings: {run.voxel_spacings}")
            if len(run.voxel_spacings) > 0:
                print(f"Tomo types in first voxel spacing: {run.voxel_spacings[0].tomograms}")
            return None, None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(tomogram.zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not tomogram:
                    return None, None
                shape = zarr.open(tomogram.zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data'], shape

    def create_ball(center, radius):
        zc, yc, xc = center
        shape = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        ball = np.zeros(shape, dtype=np.uint8)

        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - np.array([radius, radius, radius])) <= radius:
                        ball[z, y, x] = 1

        return ball

    # Function to paint picks into the segmentation as balls
    def paint_picks_as_balls(painting_seg_array, pick_location, segmentation_id, radius):
        z, y, x = pick_location
        ball = create_ball((radius, radius, radius), radius)

        # Calculate the bounding box of the ball in the segmentation array
        z_min = max(0, z - radius)
        z_max = min(painting_seg_array.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(painting_seg_array.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(painting_seg_array.shape[2], x + radius + 1)

        # Calculate the region within the ball that will be used
        z_ball_min = max(0, radius - z)
        z_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[0] - z)
        y_ball_min = max(0, radius - y)
        y_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[1] - y)
        x_ball_min = max(0, radius - x)
        x_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[2] - x)

        # Create a mask
        mask = ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max] == 1

        # Assign values directly to the numpy array using the mask
        region = painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max]
        region[mask] = segmentation_id

    # Function to paint picks into the segmentation
    def paint_picks(run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor):
        for pick in picks:
            pick_location = pick['location']
            pick_name = pick['object_type']
            segmentation_id = segmentation_mapping.get(pick_name)

            if segmentation_id is None:
                print(f"Skipping unknown object type: {pick_name}")
                continue

            # Convert the location into the appropriate voxel coordinates
            z, y, x = pick_location
            z = int(z / voxel_spacing)
            y = int(y / voxel_spacing)
            x = int(x / voxel_spacing)

            particle_radius = next(obj.radius for obj in root.config.pickable_objects if obj.name == pick_name)
            ball_radius = int(particle_radius * ball_radius_factor / voxel_spacing)

            paint_picks_as_balls(painting_seg_array, (z, y, x), segmentation_id, ball_radius)

    def process_run(run):
        painting_seg, shape = get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing)

        if painting_seg is None:
            raise ValueError(f"Unable to obtain or create painting segmentation for run '{run.name}'.")

        # Create a mapping from pick object names to segmentation IDs
        segmentation_mapping = {obj.name: obj.label for obj in root.config.pickable_objects}

        # Create an in-memory numpy array
        painting_seg_array = np.zeros(shape, dtype=np.uint16)

        # Collect all picks and paint them into the segmentation
        user_ids = set()
        session_ids = set()
        for obj in root.config.pickable_objects:
            for pick_set in run.get_picks(obj.name):
                if pick_set and pick_set.points and (not allowlist_user_ids or pick_set.user_id in allowlist_user_ids):
                    picks = [{'object_type': obj.name, 'location': (point.location.z, point.location.y, point.location.x)} for point in pick_set.points]
                    user_ids.add(pick_set.user_id)
                    session_ids.add(pick_set.session_id)
                    paint_picks(run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor)

        # Write the in-memory numpy array to the Zarr array
        painting_seg[:] = painting_seg_array

        print(f"Pickable objects: {[obj.name for obj in root.config.pickable_objects]}")
        print(f"User IDs: {user_ids}")
        print(f"Session IDs: {session_ids}")

        print(f"Painting complete for run '{run.name}'. Segmentation layers created successfully.")

    if run_name:
        run = root.get_run(run_name)
        if not run:
            raise ValueError(f"Run with name '{run_name}' not found.")
        process_run(run)
    else:
        for run in root.runs:
            process_run(run)

setup(
    group="copick",
    name="paint-from-picks",
    version="0.2.5",
    title="Paint Copick Picks into a Segmentation Layer",
    description="A solution that paints picks from a Copick project into a segmentation layer in Zarr.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "segmentation", "painting", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_name", "type": "string", "required": False, "description": "Name for the painting segmentation."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "ball_radius_factor", "type": "float", "required": True, "description": "Factor to scale the particle radius for the ball radius."},
        {"name": "run_name", "type": "string", "required": False, "description": "Name of the Copick run to process."},
        {"name": "allowlist_user_ids", "type": "string", "required": False, "description": "Comma-separated list of user IDs to include in the painting."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to use (e.g., denoised)."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
