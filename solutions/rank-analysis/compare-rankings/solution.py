###album catalog: cellcanvas

from album.runner.api import setup, get_args
import json
import os
import glob

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - pip:
    - album
    - rbo
"""

def run():
    import numpy as np
    from scipy.stats import weightedtau, rankdata
    from rbo import rbo

    args = get_args()
    json_directory = args.json_directory
    config_json = args.config_json
    beta = float(args.beta)
    weights_arg = args.weights.split(',')
    weights = {pair.split('=')[0]: float(pair.split('=')[1]) for pair in weights_arg}
    output_json = args.output_json if 'output_json' in args else None

    def list_candidate_names(json_directory):
        json_files = glob.glob(os.path.join(json_directory, "result_*.json"))
        candidate_names = [os.path.basename(f)[7:-5] for f in json_files]
        return candidate_names

    def load_results(candidate_name):
        file_path = os.path.join(json_directory, f"result_{candidate_name}.json")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)

    def compute_micro_avg_fbeta(results, runs):
        type_metrics = {}

        for run_name in runs:
            if run_name in results:
                for particle_type, metrics in results[run_name].items():
                    if particle_type not in weights:
                        continue
                    if particle_type not in type_metrics:
                        type_metrics[particle_type] = {
                            'total_tp': 0,
                            'total_fp': 0,
                            'total_fn': 0,
                            'total_reference_particles': 0,
                            'total_candidate_particles': 0
                        }
                    
                    tp = metrics['tp']
                    fp = metrics['fp']
                    fn = metrics['fn']
                    
                    type_metrics[particle_type]['total_tp'] += tp
                    type_metrics[particle_type]['total_fp'] += fp
                    type_metrics[particle_type]['total_fn'] += fn
                    type_metrics[particle_type]['total_reference_particles'] += metrics['num_reference_particles']
                    type_metrics[particle_type]['total_candidate_particles'] += metrics['num_candidate_particles']

        micro_avg_results = {}
        for particle_type, totals in type_metrics.items():
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
                'total_reference_particles': int(totals['total_reference_particles']),
                'total_candidate_particles': int(totals['total_candidate_particles']),
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return micro_avg_results

    def compute_weighted_fbeta(micro_avg_results):
        weighted_fbeta_sum = 0.0
        total_weight = sum(weights.get(particle_type, 1.0) for particle_type in micro_avg_results.keys())
        
        for particle_type, metrics in micro_avg_results.items():
            weight = weights.get(particle_type, 1.0)
            fbeta = metrics['f_beta_score']
            weighted_fbeta_sum += fbeta * weight

        aggregate_fbeta = weighted_fbeta_sum / total_weight
        return aggregate_fbeta

    def compute_rankings(candidate_names, runs):
        rankings = []
        for candidate_name in candidate_names:
            results = load_results(candidate_name)
            if results and 'all_results' in results:
                micro_avg_results = compute_micro_avg_fbeta(results['all_results'], runs)
                aggregate_fbeta = compute_weighted_fbeta(micro_avg_results)
                rankings.append((candidate_name, aggregate_fbeta))
        return rankings

    def rank_order(rankings):
        rankings.sort(key=lambda x: x[1], reverse=True)
        return [candidate_name for candidate_name, _ in rankings]

    def compute_metrics(rank_order_public, rank_order_private):
        metrics = {}
        
        public_ranks_dict = {candidate: rank for rank, candidate in enumerate(rank_order_public)}
        private_ranks_dict = {candidate: rank for rank, candidate in enumerate(rank_order_private)}

        common_candidates = list(set(rank_order_public).intersection(rank_order_private))
        public_rank_vector = [public_ranks_dict[candidate] for candidate in common_candidates]
        private_rank_vector = [private_ranks_dict[candidate] for candidate in common_candidates]

        tau, _ = weightedtau(public_rank_vector, private_rank_vector)

        rbo_score = rbo.RankingSimilarity(rank_order_public, rank_order_private).rbo()
        top_5_concordance = rbo.RankingSimilarity(rank_order_public[:5], rank_order_private[:5]).rbo()
        top_10_concordance = rbo.RankingSimilarity(rank_order_public[:10], rank_order_private[:10]).rbo()
        top_25_concordance = rbo.RankingSimilarity(rank_order_public[:25], rank_order_private[:25]).rbo()

        metrics = {
            'Weighted Kendall\'s Tau': tau,
            'Rank Biased Overlap': rbo_score,
            'Top-5 Concordance': top_5_concordance,
            'Top-10 Concordance': top_10_concordance,
            'Top-25 Concordance': top_25_concordance
        }
        
        return metrics

    with open(config_json, 'r') as f:
        config = json.load(f)

    training_runs = config['training_runs']
    public_test_runs = config['public_test_runs']
    private_test_runs = config['private_test_runs']

    candidate_names = list_candidate_names(json_directory)

    public_test_rankings = compute_rankings(candidate_names, public_test_runs)
    private_test_rankings = compute_rankings(candidate_names, private_test_runs)

    public_test_rank_order = rank_order(public_test_rankings)
    private_test_rank_order = rank_order(private_test_rankings)

    metrics = compute_metrics(public_test_rank_order, private_test_rank_order)

    results = {
        'metrics': metrics,
        'public_test_top_25': public_test_rank_order[:25],
        'private_test_top_25': private_test_rank_order[:25]
    }

    if output_json:
        print(f"Saving results to {output_json}")
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

setup(
    group="rank-analysis",
    name="compare-rankings",
    version="0.0.13",
    title="Compare Rankings from Different Runs",
    description="A solution that compares the rankings of candidates in the public and private test sets using various rank metrics.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "rankings", "comparison"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "json_directory", "type": "string", "required": True, "description": "Directory containing the JSON files with results."},
        {"name": "config_json", "type": "string", "required": True, "description": "Path to the configuration JSON file with run names."},
        {"name": "beta", "type": "string", "required": True, "description": "Beta value for the f-beta score."},
        {"name": "weights", "type": "string", "required": True, "description": "Comma-separated string of weights for each particle type (e.g., type1=0.5,type2=1.0)."},
        {"name": "output_json", "type": "string", "required": False, "description": "Path to save the output JSON file with the results."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
