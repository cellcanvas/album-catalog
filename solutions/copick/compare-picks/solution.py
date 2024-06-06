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
            return np.inf, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0
        
        ref_tree = cKDTree(reference_points)
        distances, indices = ref_tree.query(candidate_points)
        
        valid_distances = distances[distances != np.inf]
        average_distance = np.mean(valid_distances) if valid_distances.size > 0 else np.inf
        
        matches_within_threshold = distances <= threshold
        
        # Precision: proportion of correctly identified candidates
        precision = np.sum(matches_within_threshold) / len(candidate_points)
        
        # Recall: proportion of reference points correctly identified
        unique_matched_indices = np.unique(indices[matches_within_threshold])
        recall = len(unique_matched_indices) / len(reference_points)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        num_reference_particles = len(reference_points)
        num_candidate_particles = len(candidate_points)
        num_matched_particles = np.sum(matches_within_threshold)
        percent_matched_reference = (len(unique_matched_indices) / num_reference_particles) * 100
        percent_matched_candidate = (num_matched_particles / num_candidate_particles) * 100
        
        return (average_distance, precision, recall, f1, num_reference_particles, 
                num_candidate_particles, len(unique_matched_indices), percent_matched_reference, 
                percent_matched_candidate)

    reference_picks = load_picks(run, reference_user_id, reference_session_id)
    candidate_picks = load_picks(run, candidate_user_id, candidate_session_id)

    # TODO remove this after the current analysis [2024 / 06 / 06]
    # Swap x and z coordinates for reference picks
    for k in self.reference_picks.keys():
        self.reference_picks[k] = self.reference_picks[k][:, [2, 1, 0]]

    
    results = {}
    for particle_type in reference_picks:
        if particle_type in candidate_picks:
            (avg_distance, precision, recall, f1, num_reference, num_candidate, num_matched, 
             percent_matched_ref, percent_matched_cand) = compute_metrics(
                reference_picks[particle_type],
                candidate_picks[particle_type],
                distance_threshold
            )
            results[particle_type] = {
                'average_distance': avg_distance,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'num_reference_particles': num_reference,
                'num_candidate_particles': num_candidate,
                'num_matched_particles': num_matched,
                'percent_matched_reference': percent_matched_ref,
                'percent_matched_candidate': percent_matched_cand
            }
        else:
            results[particle_type] = {
                'average_distance': np.inf,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'num_reference_particles': len(reference_picks[particle_type]),
                'num_candidate_particles': 0,
                'num_matched_particles': 0,
                'percent_matched_reference': 0.0,
                'percent_matched_candidate': 0.0
            }

    for particle_type, metrics in results.items():
        print(f"Particle: {particle_type}")
        print(f"  Average Distance: {metrics['average_distance']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F1 Score: {metrics['f1_score']}")
        print(f"  Number of Reference Particles: {metrics['num_reference_particles']}")
        print(f"  Number of Candidate Particles: {metrics['num_candidate_particles']}")
        print(f"  Number of Matched Particles: {metrics['num_matched_particles']}")
        print(f"  Percent Matched (Reference): {metrics['percent_matched_reference']}%")
        print(f"  Percent Matched (Candidate): {metrics['percent_matched_candidate']}%")

setup(
    group="copick",
    name="compare-picks",
    version="0.0.12",
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
