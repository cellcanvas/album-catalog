###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - zarr
  - pandas
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import json
    import numpy as np
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    def fetch_embedding(tomo, feature_types, location, radius):
        embeddings = {}
        spatial_coords = np.round(np.array([location.x, location.y, location.z])).astype(int)
        spatial_coords = np.clip(spatial_coords, [0, 0, 0], np.array(zarr.open(tomo.zarr(), "r")["0"].shape) - 1)

        for feature_type in feature_types:
            feature_data = tomo.get_features(feature_type)
            if feature_data is not None:
                dataset = zarr.open(feature_data.zarr(), mode='r')
                x_range = slice(max(0, spatial_coords[0] - radius), min(dataset.shape[1], spatial_coords[0] + radius + 1))
                y_range = slice(max(0, spatial_coords[1] - radius), min(dataset.shape[2], spatial_coords[1] + radius + 1))
                z_range = slice(max(0, spatial_coords[2] - radius), min(dataset.shape[3], spatial_coords[2] + radius + 1))
                embedding = np.median(dataset[:, x_range, y_range, z_range], axis=(1, 2, 3))
                embeddings[feature_type] = embedding
            else:
                embeddings[feature_type] = None
        return embeddings

    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    session_id = args.session_id
    new_session_id = args.new_session_id
    feature_types = args.feature_types.split(',')
    radius = int(args.radius)
    tomo_type = args.tomo_type
    voxel_spacing = int(args.voxel_spacing)
    median_embeddings_path = args.median_embeddings_path

    # Load the Copick root configuration
    root = CopickRootFSSpec.from_file(copick_config_path)
    run = root.get_run(run_name)
    
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    # Load the median embeddings
    with open(median_embeddings_path, 'r') as f:
        median_embeddings = json.load(f)

    # Process picks
    picks = run.get_picks(session_id=session_id)
    if not picks:
        print(f"No picks found for session_id {session_id} in run {run_name}")
        return

    # Get tomogram for embedding calculation
    tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
    if tomo is None:
        raise ValueError(f"No tomogram found for type '{tomo_type}' and voxel spacing '{voxel_spacing}'.")

    for pick in picks:
        object_type = pick.pickable_object_name
        if object_type not in median_embeddings:
            print(f"No median embedding found for object type '{object_type}'")
            continue
        
        median_emb = np.array(median_embeddings[object_type])
        
        # Create a new pick set with the new session ID
        new_pick_set = run.new_picks(object_type, new_session_id, pick.user_id)

        new_points = []
        for point in pick.points:
            # Fetch embedding for the point's location
            location = point.location
            point_embeddings = fetch_embedding(tomo, feature_types, location, radius)
            
            # Combine embeddings for distance calculation
            combined_embedding = []
            for feature_type, embedding in point_embeddings.items():
                if embedding is not None:
                    combined_embedding.extend(embedding)

            if combined_embedding:
                combined_embedding = np.array(combined_embedding)
                # Compute the distance to the median embedding
                distance = np.linalg.norm(combined_embedding - median_emb)
            else:
                distance = None
            
            # Create a new point with the same location and the computed score
            new_point = CopickPoint(location=location)
            new_point.score = distance
            new_points.append(new_point)

        # Assign the new points to the new pick set and store it
        new_pick_set.points = new_points
        new_pick_set.store()

    print(f"Created new picks with scores for session_id {new_session_id} in run {run_name}")

setup(
    group="copick",
    name="pick-distance-from-embedding",
    version="0.0.5",
    title="Create Picks with Distance to Median Embedding",
    description="Creates a new set of picks for a new session ID, containing the same locations but including the distance to the median embedding in the 'score' attribute.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "picks", "embedding", "cryoet"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID to filter picks."},
        {"name": "new_session_id", "type": "string", "required": True, "description": "New session ID for the newly created picks."},
        {"name": "feature_types", "type": "string", "required": True, "description": "Comma-separated list of feature types to extract embeddings for."},
        {"name": "radius", "type": "integer", "required": True, "description": "Radius for calculating median embeddings."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram, e.g. denoised."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "median_embeddings_path", "type": "string", "required": True, "description": "Path to the JSON file containing median embeddings."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
