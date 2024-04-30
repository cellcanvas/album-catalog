###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - pandas
  - numpy
  - pip:
    - album
    - dill
"""

def run():
    import os
    import numpy as np
    import pandas as pd
    import zarr

    def load_embeddings(zarr_directory):
        return zarr.open(zarr_directory, mode='r')

    def process_box(embedding_dataset, box_slice, median_embeddings_df, distance_threshold):
        # Fetch all embeddings for the slice
        embeddings = embedding_dataset[box_slice].compute()

        # Initialize list to collect matches
        matches = []

        # Loop through each class median to compare
        for index, row in median_embeddings_df.iterrows():
            class_median = row.filter(regex='^median_emb_').values
            # Compute the difference array for this class median
            diff = embeddings - class_median.reshape(-1, 1, 1, 1)
            distances = np.linalg.norm(diff, axis=0)
            # Find locations where the distance is within the threshold
            within_threshold = distances < (distance_threshold * row['median_distance'])
            match_indices = np.argwhere(within_threshold)

            # Adjust match indices to fit the absolute coordinates within the dataset
            match_indices[:, 0] += box_slice[1].start
            match_indices[:, 1] += box_slice[2].start
            match_indices[:, 2] += box_slice[3].start

            # Append matches to the list
            for coords in match_indices:
                matches.append((index, *coords, distances[tuple(coords)]))

        return matches

    args = get_args()
    embedding_directory = args.embedding_directory
    median_embeddings_path = args.median_embeddings_path
    matches_output_path = args.matches_output_path
    distance_threshold = args.distance_threshold

    embedding_dataset = load_embeddings(embedding_directory)
    median_embeddings_df = pd.read_csv(median_embeddings_path)

    # Define the box size
    box_size = 25
    shape = embedding_dataset.shape[1:]  # ignore the embedding dimension in shape
    matches = []
    
    num_boxes = len(list(range(0, shape[2], box_size))) * len(list(range(0, shape[1], box_size))) * len(list(range(0, shape[0], box_size)))
    print(f"Number of boxes to check: {num_boxes}")
    boxes_checked = 0    
    for z in range(0, shape[2], box_size):
        for y in range(0, shape[1], box_size):
            for x in range(0, shape[0], box_size):
                box_slice = (
                    slice(None),  # all embedding dimensions
                    slice(x, min(x + box_size, shape[0])),
                    slice(y, min(y + box_size, shape[1])),
                    slice(z, min(z + box_size, shape[2]))
                )
                results = process_box(embedding_dataset, box_slice, median_embeddings_df, distance_threshold)
                matches.extend(results)
                print(f"Processed box {boxes_checked} had {len(results)} hits")
                boxes_checked += 1

    matches_df = pd.DataFrame(matches, columns=['class', 'x', 'y', 'z', 'distance'])
    matches_df.to_csv(matches_output_path, index=False)
    print(f"Matches saved to {matches_output_path}")
    
setup(
    group="copick",
    name="discover-picks",
    version="0.0.13",
    title="Classify and Match Embeddings to Known Particle Classes with Multithreading",
    description="Uses multithreading to compare median embeddings from a Zarr dataset to known class medians and identifies matches based on a configurable distance threshold.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "embedding", "classification", "cryoet", "multithreading"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "embedding_directory", "type": "string", "required": True, "description": "Path to the embedding Zarr directory."},
        {"name": "median_embeddings_path", "type": "string", "required": True, "description": "Path to the CSV file with median embeddings and distances."},
        {"name": "matches_output_path", "type": "string", "required": True, "description": "Path for the output file containing matches."},
        {"name": "distance_threshold", "type": "float", "required": True, "description": "Distance threshold factor to consider a match."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
