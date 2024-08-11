###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - pandas
  - numpy
  - pip:
    - album
"""

def run():
    import os
    import numpy as np
    import pandas as pd
    import zarr

    def load_embeddings(embedding_directories):
        datasets = []
        for dir_path in embedding_directories:
            datasets.append(zarr.open(dir_path, mode='r'))
        return datasets

    def aggregate_embeddings(datasets, box_slice):
        # Collect embeddings from all datasets for the given box_slice
        box_embeddings = [dataset[box_slice] for dataset in datasets]
        # Stack the embeddings along the new 0-axis
        stacked_embeddings = np.concatenate(box_embeddings, axis=0)
        return stacked_embeddings


    def process_box(combined_medians, median_embeddings_df, distance_threshold):
        matches = []
        for index, row in median_embeddings_df.iterrows():
            class_median = row.filter(regex='^median_emb_').values
            diff = (combined_medians - class_median.reshape(-1, 1, 1, 1)).astype(float)
            distances = np.linalg.norm(diff, axis=0)
            within_threshold = distances < (distance_threshold * row['median_distance'])
            match_indices = np.argwhere(within_threshold)

            for match in match_indices:
                global_coords = match  # Directly use match since it's global within the box
                distance = distances[tuple(match)]
                matches.append((index, *global_coords, distance))

        return matches

    args = get_args()
    embedding_directories = args.embedding_directories.split(',')
    median_embeddings_path = args.median_embeddings_path
    matches_output_path = args.matches_output_path
    distance_threshold = args.distance_threshold

    embedding_datasets = load_embeddings(embedding_directories)
    median_embeddings_df = pd.read_csv(median_embeddings_path)

    box_size = 50
    matches = []
    boxes_checked = 0

    shape = embedding_datasets[0].shape[1:]  # Assuming all datasets have the same shape
    num_boxes = len(list(range(0, shape[2], box_size))) * len(list(range(0, shape[1], box_size))) * len(list(range(0, shape[0], box_size)))
    print(f"Number of boxes to check: {num_boxes}")

    for z in range(0, shape[2], box_size):
        for y in range(0, shape[1], box_size):
            for x in range(0, shape[0], box_size):
                box_slice = (
                    slice(None),  # all embedding dimensions
                    slice(x, min(x + box_size, shape[0])),
                    slice(y, min(y + box_size, shape[1])),
                    slice(z, min(z + box_size, shape[2]))
                )
                combined_medians = aggregate_embeddings(embedding_datasets, box_slice)
                results = process_box(combined_medians, median_embeddings_df, distance_threshold)
                matches.extend(results)
                print(f"Processed box {boxes_checked} had {len(results)} hits")
                boxes_checked += 1

    matches_df = pd.DataFrame(matches, columns=['class', 'x', 'y', 'z', 'distance'])
    matches_df.to_csv(matches_output_path, index=False)
    print(f"Matches saved to {matches_output_path}")    

setup(
    group="copick",
    name="discover-picks",
    version="0.1.4",
    title="Classify and Match Embeddings to Known Particle Classes with Multithreading Across Multiple Directories",
    description="Uses multithreading to compare median embeddings from multiple Zarr datasets to known class medians and identifies matches based on a configurable distance threshold.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "embedding", "classification", "cryoet", "multithreading"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "embedding_directories", "type": "string", "required": True, "description": "Paths to the embedding Zarr directories separated by commas."},
        {"name": "median_embeddings_path", "type": "string", "required": True, "description": "Path to the CSV file with median embeddings and distances."},
        {"name": "matches_output_path", "type": "string", "required": True, "description": "Path for the output file containing matches."},
        {"name": "distance_threshold", "type": "float", "required": True, "description": "Distance threshold factor to consider a match."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)

