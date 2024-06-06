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
  - scipy
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import numpy as np
    from scipy.spatial import cKDTree
    from sklearn.metrics import precision_score, recall_score, f1_score
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    args = get_args()
    copick_config_path = args.copick_config_path
    reference_user_id = args.reference_user_id
    reference_session_id = args.reference_session_id
    candidate_user_id = args.candidate_user_id
    candidate_session_id = args.candidate_session_id
    distance_threshold = float(args.distance_threshold)
    run_name = args.run_name

    root = CopickRootFSSpec.from_file(copick_config_path)
    run = root.get_run(run_name)

    def load_picks(run, user_id, session_id):
        picks = run.get_picks(user_id=user_id, session_id=session_id)
        pick_points = {}
        for pick in picks:
            points = pick.points
            object_name = pick.pickable_object_name
            pick_points[object_name] = np.array([[p.location.x, p.location.y, p.location.z] for p in points])
        return pick_points

    def compute_metrics(reference_points, candidate_points, threshold):
        if len(candidate_points) == 0:
            return np.inf, 0.0, 0.0, 0.0
        
        ref_tree = cKDTree(reference_points)
        distances, indices = ref_tree.query(candidate_points, distance_upper_bound=threshold)
        
        valid_distances = distances[distances != np.inf]
        average_distance = np.mean(valid_distances) if valid_distances.size > 0 else np.inf
        
        matches = distances != np.inf
        y_true = np.ones(len(reference_points))
        y_pred = np.zeros(len(reference_points))
        if matches.any():
            y_pred[indices[matches] < len(reference_points)] = 1
        
        precision = precision_score(np.ones(len(candidate_points)), matches, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return average_distance, precision, recall, f1

    reference_picks = load_picks(run, reference_user_id, reference_session_id)
    candidate_picks = load_picks(run, candidate_user_id, candidate_session_id)
    
    results = {}
    for particle_type in reference_picks:
        if particle_type in candidate_picks:
            avg_distance, precision, recall, f1 = compute_metrics(
                reference_picks[particle_type],
                candidate_picks[particle_type],
                distance_threshold
            )
            results[particle_type] = {
                'average_distance': avg_distance,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            results[particle_type] = {
                'average_distance': np.inf,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

    for particle_type, metrics in results.items():
        print(f"Particle: {particle_type}")
        print(f"  Average Distance: {metrics['average_distance']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F1 Score: {metrics['f1_score']}")

setup(
    group="copick",
    name="compare-picks",
    version="0.0.6",
    title="Compare Picks from Different Users and Sessions",
    description="A solution that compares the picks from a reference user and session to a candidate user and session for all particle types, providing metrics like average distance, precision, recall, and F1 score.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "picks", "comparison", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "reference_user_id", "type": "string", "required": True, "description": "User ID for the reference picks."},
        {"name": "reference_session_id", "type": "string", "required": True, "description": "Session ID for the reference picks."},
        {"name": "candidate_user_id", "type": "string", "required": True, "description": "User ID for the candidate picks."},
        {"name": "candidate_session_id", "type": "string", "required": True, "description": "Session ID for the candidate picks."},
        {"name": "distance_threshold", "type": "float", "required": True, "description": "Distance threshold for matching points in Angstrom."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
