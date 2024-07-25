###album catalog: cellcanvas

from album.runner.api import setup, get_args
import json
import os

env_file = """
channels:
  - conda-forge
  - defaults
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
    training_runs = args.training_runs.split(',')
    public_test_runs = args.public_test_runs.split(',')
    private_test_runs = args.private_test_runs.split(',')
    output_json = args.output_json if 'output_json' in args else None

    def load_results(run_name):
        file_path = os.path.join(json_directory, f"{run_name}.json")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)

    def compute_rankings(run_names):
        rankings = {}
        for run_name in run_names:
            results = load_results(run_name)
            if results is not None:
                for particle_type, metrics in results['micro_avg_results'].items():
                    if particle_type not in rankings:
                        rankings[particle_type] = []
                    fbeta = metrics['f_beta_score']
                    rankings[particle_type].append((run_name, fbeta))
        return rankings

    def rank_order(rankings):
        rank_orders = {}
        for particle_type, scores in rankings.items():
            scores.sort(key=lambda x: x[1], reverse=True)
            rank_orders[particle_type] = [run_name for run_name, _ in scores]
        return rank_orders

    def compute_metrics(rank_order_public, rank_order_private):
        metrics = {}
        for particle_type in rank_order_public.keys():
            if particle_type in rank_order_private:
                public_ranks = rank_order_public[particle_type]
                private_ranks = rank_order_private[particle_type]
                public_ranks_dict = {run: rank for rank, run in enumerate(public_ranks)}
                private_ranks_dict = {run: rank for rank, run in enumerate(private_ranks)}

                common_runs = list(set(public_ranks).intersection(private_ranks))
                public_rank_vector = [public_ranks_dict[run] for run in common_runs]
                private_rank_vector = [private_ranks_dict[run] for run in common_runs]

                tau, _ = weightedtau(public_rank_vector, private_rank_vector)

                rbo_score = rbo.RankingSimilarity(public_ranks, private_ranks).rbo()
                top_5_concordance = rbo.RankingSimilarity(public_ranks[:5], private_ranks[:5]).rbo()
                top_10_concordance = rbo.RankingSimilarity(public_ranks[:10], private_ranks[:10]).rbo()
                top_25_concordance = rbo.RankingSimilarity(public_ranks[:25], private_ranks[:25]).rbo()

                metrics[particle_type] = {
                    'Weighted Kendall\'s Tau': tau,
                    'Rank Biased Overlap': rbo_score,
                    'Top-5 Concordance': top_5_concordance,
                    'Top-10 Concordance': top_10_concordance,
                    'Top-25 Concordance': top_25_concordance
                }
        return metrics

    training_rankings = compute_rankings(training_runs)
    public_test_rankings = compute_rankings(public_test_runs)
    private_test_rankings = compute_rankings(private_test_runs)

    public_test_rank_order = rank_order(public_test_rankings)
    private_test_rank_order = rank_order(private_test_rankings)

    metrics = compute_metrics(public_test_rank_order, private_test_rank_order)

    if output_json:
        print(f"Saving results to {output_json}")
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))

setup(
    group="rank-analysis",
    name="compare-rankings",
    version="0.0.1",
    title="Compare Rankings from Different Runs",
    description="A solution that compares the rankings of candidates in the public and private test sets using various rank metrics.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "rankings", "comparison"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "json_directory", "type": "string", "required": True, "description": "Directory containing the JSON files with results."},
        {"name": "training_runs", "type": "string", "required": True, "description": "Comma-separated list of training run names."},
        {"name": "public_test_runs", "type": "string", "required": True, "description": "Comma-separated list of public test run names."},
        {"name": "private_test_runs", "type": "string", "required": True, "description": "Comma-separated list of private test run names."},
        {"name": "output_json", "type": "string", "required": False, "description": "Path to save the output JSON file with the results."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
