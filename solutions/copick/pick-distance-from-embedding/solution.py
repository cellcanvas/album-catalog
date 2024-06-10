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
    import os
    import json
    import numpy as np
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    session_id = args.session_id
    new_session_id = args.new_session_id
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

    for pick in picks:
        object_type = pick.pickable_object_name
        if object_type not in median_embeddings:
            print(f"No median embedding found for object type '{object_type}'")
            continue
        
        median_emb = np.array(median_embeddings[object_type])
        
        # Create a new pick set with the new session ID
        new_pick_set = run.new_picks(object_type, new_session_id, pick.user_id)

        for point in pick.points:
            # Compute the distance to the median embedding
            embedding = point.get_embedding()
            if embedding is not None:
                embedding = np.array(embedding)
                distance = np.linalg.norm(embedding - median_emb)
            else:
                distance = None
            
            # Create a new point with the same location and the computed score
            new_point = CopickPoint(location=point.location)
            new_point.score = distance
            new_pick_set.add_point(new_point)

        # Store the new pick set
        new_pick_set.store()

    print(f"Created new picks with scores for session_id {new_session_id} in run {run_name}")

setup(
    group="copick",
    name="pick-distance-from-embedding",
    version="0.0.1",
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
        {"name": "median_embeddings_path", "type": "string", "required": True, "description": "Path to the JSON file containing median embeddings."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
