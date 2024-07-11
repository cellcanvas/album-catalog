###album catalog: cellcanvas

from album.runner.api import setup, get_args
import json

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
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    args = get_args()
    copick_config_path = args.copick_config_path
    reference_user_id = args.reference_user_id
    reference_session_id = args.reference_session_id
    candidate_user_id = args.candidate_user_id
    candidate_session_id = args.candidate_session_id
    distance_multiplier = float(args.distance_threshold)
    beta = float(args.beta)
    provided_run_name = args.run_name
    output_json = args.output_json if 'output_json' in args else None
    weights = json.loads(args.weights) if args.weights else {}

    root = CopickRootFSSpec.from_file(copick_config_path)

    # Extract pickable objects and their radii
    pickable_objects = {obj.name: obj.radius for obj in root.pickable_objects}

    def load_picks(run, user_id, session_id):
        print(f"Loading picks for user {user_id}, session {session_id}")
        picks = run.get_picks(user_id=user_id, session_id=session_id)
        pick_points = {}
        for pick in picks:
            points = pick.points
            object_name = pick.pickable_object_name
            radius = pickable_objects.get(object_name, None)
            if radius is None:
                print(f"Skipping object {object_name} as it has no radius.")
                continue
            pick_points[object_name] = {
                'points': np.array([[p.location.x, p.location.y, p.location.z] for p in points]),
                'radius': radius
            }
            print(f"Loaded {len(points)} points for object {object_name} with radius {radius}")
        return pick_points

    def compute_metrics(reference_points, reference_radius, candidate_points, distance_multiplier, beta):
        print(f"Computing metrics with {len(reference_points)} reference points and {len(candidate_points)} candidate points")
        
        if len(reference_points) == 0:
            print("No reference points, returning default metrics for no reference points")
            return np.inf, 0.0, 0.0, 0.0, 0, len(candidate_points), 0, 0.0, 0.0
        
        if len(candidate_points) == 0:
            print("No candidate points, returning default metrics for no candidate points")
            return np.inf, 0.0, 0.0, 0.0, len(reference_points), 0, 0, 0.0, 0.0
        
        ref_tree = cKDTree(reference_points)
        threshold = reference_radius * distance_multiplier
        distances, indices = ref_tree.query(candidate_points)
        
        valid_distances = distances[distances != np.inf]
        average_distance = np.mean(valid_distances) if valid_distances.size > 0 else np.inf
        
        matches_within_threshold = distances <= threshold
        
        precision = np.sum(matches_within_threshold) / len(candidate_points)
        
        unique_matched_indices = np.unique(indices[matches_within_threshold])
        recall = len(unique_matched_indices) / len(reference_points)
        
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        
        num_reference_particles = len(reference_points)
        num_candidate_particles = len(candidate_points)
        num_matched_particles = np.sum(matches_within_threshold)
        percent_matched_reference = (len(unique_matched_indices) / num_reference_particles) * 100
        percent_matched_candidate = (num_matched_particles / num_candidate_particles) * 100
        
        return (average_distance, precision, recall, fbeta, num_reference_particles, 
                num_candidate_particles, len(unique_matched_indices), percent_matched_reference, 
                percent_matched_candidate)

    def process_run(run):
        reference_picks = load_picks(run, reference_user_id, reference_session_id)
        candidate_picks = load_picks(run, candidate_user_id, candidate_session_id)
        
        results = {}
        for particle_type in reference_picks:
            if particle_type in candidate_picks:
                (avg_distance, precision, recall, fbeta, num_reference, num_candidate, num_matched, 
                 percent_matched_ref, percent_matched_cand) = compute_metrics(
                    reference_picks[particle_type]['points'],
                    reference_picks[particle_type]['radius'],
                    candidate_picks[particle_type]['points'],
                    distance_multiplier,
                    beta
                )
                results[particle_type] = {
                    'average_distance': avg_distance,
                    'precision': precision,
                    'recall': recall,
                    'f_beta_score': fbeta,
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
                    'f_beta_score': 0.0,
                    'num_reference_particles': len(reference_picks[particle_type]['points']),
                    'num_candidate_particles': 0,
                    'num_matched_particles': 0,
                    'percent_matched_reference': 0.0,
                    'percent_matched_candidate': 0.0
                }
        return results

    all_results = {}
    runs = [root.get_run(provided_run_name)] if provided_run_name else root.runs
    
    for run in runs:
        run_name = run.name
        print(f"Processing run: {run_name}")
        results = process_run(run)
        all_results[run_name] = results

    micro_avg_results = {}
    aggregate_fbeta = 0.0
    total_weight = 0.0
    
    if not provided_run_name:
        type_metrics = {}

        for run_results in all_results.values():
            for particle_type, metrics in run_results.items():
                weight = weights.get(particle_type, 1.0)
                total_weight += weight
                
                if particle_type not in type_metrics:
                    type_metrics[particle_type] = {
                        'total_tp': 0,
                        'total_fp': 0,
                        'total_fn': 0,
                        'total_reference_particles': 0,
                        'total_candidate_particles': 0
                    }
                
                tp = metrics['num_matched_particles']
                fp = metrics['num_candidate_particles'] - tp
                fn = metrics['num_reference_particles'] - tp
                
                type_metrics[particle_type]['total_tp'] += tp
                type_metrics[particle_type]['total_fp'] += fp
                type_metrics[particle_type]['total_fn'] += fn
                type_metrics[particle_type]['total_reference_particles'] += metrics['num_reference_particles']
                type_metrics[particle_type]['total_candidate_particles'] += metrics['num_candidate_particles']
        
        for particle_type, totals in type_metrics.items():
            weight = weights.get(particle_type, 1.0)
            tp = totals['total_tp']
            fp = totals['total_fp']
            fn = totals['total_fn']
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
            
            micro_avg_results[particle_type] = {
                'precision': precision,
                'recall': recall,
                'f_beta_score': fbeta,
                'total_reference_particles': totals['total_reference_particles'],
                'total_candidate_particles': totals['total_candidate_particles']
            }
            aggregate_fbeta += fbeta * weight

        aggregate_fbeta /= total_weight
        print(f"Aggregate F-beta Score: {aggregate_fbeta} (beta={beta})")
        print("Micro-averaged metrics across all runs per particle type:")
        for particle_type, metrics in micro_avg_results.items():
            print(f"Particle: {particle_type}")
            print(f"  Precision: {metrics['precision']}")
            print(f"  Recall: {metrics['recall']}")
            print(f"  F-beta Score: {metrics['f_beta_score']} (beta={beta})")
            print(f"  Total Reference Particles: {metrics['total_reference_particles']}")
            print(f"  Total Candidate Particles: {metrics['total_candidate_particles']}")
    else:
        micro_avg_results = all_results[provided_run_name]
        for particle_type, metrics in micro_avg_results.items():
            print(f"Particle: {particle_type}")
            print(f"  Average Distance: {metrics['average_distance']}")
            print(f"  Precision: {metrics['precision']}")
            print(f"  Recall: {metrics['recall']}")
            print(f"  F-beta Score: {metrics['f_beta_score']} (beta={beta})")
            print(f"  Number of Reference Particles: {metrics['num_reference_particles']}")
            print(f"  Number of Candidate Particles: {metrics['num_candidate_particles']}")
            print(f"  Number of Matched Particles: {metrics['num_matched_particles']}")
            print(f"  Percent Matched (Reference): {metrics['percent_matched_reference']}%")
            print(f"  Percent Matched (Candidate): {metrics['percent_matched_candidate']}%")

    if output_json:
        print(f"Saving results to {output_json}")
        with open(output_json, 'w') as f:
            json.dump({
                'micro_avg_results': micro_avg_results,
                'aggregate_fbeta': aggregate_fbeta
            }, f, indent=4)


setup(
    group="copick",
    name="compare-picks",
    version="0.0.25",
    title="Compare Picks from Different Users and Sessions with F-beta Score",
    description="A solution that compares the picks from a reference user and session to a candidate user and session for all particle types, providing metrics like average distance, precision, recall, and F-beta score. Computes micro-averaged F-beta score across all runs if run_name is not provided.",
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
        {"name": "distance_threshold", "type": "float", "required": True, "description": "Distance threshold multiplier for matching points (e.g., 1.5x the radius as default)."},
        {"name": "beta", "type": "float", "required": True, "description": "Beta value for the F-beta score."},
        {"name": "run_name", "type": "string", "required": False, "description": "Name of the Copick run to process. If not specified all runs will be processed."},
        {"name": "output_json", "type": "string", "required": False, "description": "Path to save the output JSON file with the results."},
        {"name": "weights", "type": "string", "required": False, "description": "JSON string with weights for each particle type."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
