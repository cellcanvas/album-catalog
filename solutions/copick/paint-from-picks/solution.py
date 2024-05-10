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
  - pip:
    - album
    - copick
"""

def run():
    import numpy as np
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = args.voxel_spacing
    ball_radius = args.ball_radius
    run_name = args.run_name

    # Load the Copick root from the configuration file
    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "painting_segmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
        if not run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised"):
            return None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised"):
                    return None
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

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

        # Paint the ball into the segmentation array
        painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max][
            ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max] == 1
        ] = segmentation_id

    # Function to paint picks into the segmentation
    def paint_picks(run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius):
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

            paint_picks_as_balls(painting_seg_array, (z, y, x), segmentation_id, ball_radius)

    # Retrieve the specified run by name
    run = root.get_run(run_name)
    if not run:
        raise ValueError(f"Run with name '{run_name}' not found.")

    print(f"Painting run '{run_name}': {run}")

    painting_seg = get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing)

    # Create a mapping from pick object names to segmentation IDs
    segmentation_mapping = {obj.name: obj.label for obj in root.config.pickable_objects}

    # Collect all picks and paint them into the segmentation
    for obj in root.config.pickable_objects:
        for pick_set in run.get_picks(obj.name, user_id=user_id):
            if pick_set and pick_set.points:
                picks = [{'object_type': obj.name, 'location': (point.location.z, point.location.y, point.location.x)} for point in pick_set.points]
                paint_picks(run, painting_seg, picks, segmentation_mapping, voxel_spacing, ball_radius)

    print(f"Painting complete. Segmentation layers created successfully.")

setup(
    group="copick",
    name="paint-from-picks",
    version="0.1.0",
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
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "ball_radius", "type": "integer", "required": True, "description": "Radius of the ball used to paint picks into the segmentation."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
