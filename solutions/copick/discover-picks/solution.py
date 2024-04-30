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
    from concurrent.futures import ProcessPoolExecutor
    import dill
    import multiprocessing

    # Set multiprocessing context
    multiprocessing.context._default_context = multiprocessing.get_context("fork")
    multiprocessing.util._ForkingPickler = dill.Pickler

    def load_embeddings(zarr_directory):
        return zarr.open(zarr_directory, mode='r')

    def worker_process(data):
        """Deserialize and execute the function."""
        func, args = dill.loads(data)
        return func(*args)

    def median_embedding(embedding_dataset, box_slice):
        # Compute the median embedding within the specified box slice
        return np.median(embedding_dataset[box_slice], axis=(1, 2, 3))
    
    def min_max_embedding(embedding_dataset, box_slice):
        # Compute the min and max embedding values within the specified box slice
        min_emb = np.min(embedding_dataset[box_slice], axis=(1, 2, 3))
        max_emb = np.max(embedding_dataset[box_slice], axis=(1, 2, 3))
        return min_emb, max_emb

    def process_location(embedding_dataset, location, median_embeddings_df, distance_threshold):
        # Process each location individually
        matches = []
        current_embedding = median_embedding(embedding_dataset, (slice(None), slice(location[0], location[0]+1), slice(location[1], location[1]+1), slice(location[2], location[2]+1)))
        for index, row in median_embeddings_df.iterrows():
            class_median = row.filter(regex='^median_emb_').values
            distance = np.linalg.norm(current_embedding - class_median)
            if distance < distance_threshold * row['median_distance']:
                matches.append((index, location[0], location[1], location[2], distance))
        return matches
    
    def process_box(embedding_dataset, box_slice, median_embeddings_df, distance_threshold):
        # Evaluate the min-max range to determine if further search is needed
        min_emb, max_emb = min_max_embedding(embedding_dataset, box_slice)
        results = []
        for row in median_embeddings_df.itertuples():
            class_median = getattr(row, 'median_emb')
            min_distance = np.linalg.norm(min_emb - class_median)
            max_distance = np.linalg.norm(max_emb - class_median)
        
            if min_distance < distance_threshold * row.median_distance or max_distance < distance_threshold * row.median_distance:
                # If within threshold, process all locations within the box
                for z in range(box_slice[3].start, box_slice[3].stop):
                    for y in range(box_slice[2].start, box_slice[2].stop):
                        for x in range(box_slice[1].start, box_slice[1].stop):
                            results.extend(process_location(embedding_dataset, (x, y, z), median_embeddings_df, distance_threshold))
        return results

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
                matches.extend(process_box(embedding_dataset, box_slice, median_embeddings_df, distance_threshold))                
                print(f"Processed box {boxes_checked}")
                boxes_checked += 1

    matches_df = pd.DataFrame(matches, columns=['class', 'x', 'y', 'z', 'distance'])
    matches_df.to_csv(matches_output_path, index=False)
    print(f"Matches saved to {matches_output_path}")
    
setup(
    group="copick",
    name="discover-picks",
    version="0.0.10",
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
