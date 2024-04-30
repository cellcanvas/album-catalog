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
"""

def run():
    import os
    import numpy as np
    import pandas as pd
    import zarr
    from concurrent.futures import ProcessPoolExecutor

    def load_embeddings(zarr_directory):
        return zarr.open(zarr_directory, mode='r')

    def median_embedding(embedding_dataset, batch_locations, radius=3):
        results = []
        for location in batch_locations:
            spatial_coords = np.clip(location, [0, 0, 0], np.array(embedding_dataset.shape[1:4]) - 1)
            x_range = slice(max(0, spatial_coords[0]-radius), min(embedding_dataset.shape[1], spatial_coords[0]+radius+1))
            y_range = slice(max(0, spatial_coords[1]-radius), min(embedding_dataset.shape[2], spatial_coords[1]+radius+1))
            z_range = slice(max(0, spatial_coords[2]-radius), min(embedding_dataset.shape[3], spatial_coords[2]+radius+1))
            median = np.median(embedding_dataset[:, x_range, y_range, z_range], axis=(1, 2, 3))
            results.append((location, median))
        return results

    def match_embeddings(class_medians, current_embeddings, distance_threshold):
        matches = []
        for location, embedding in current_embeddings:
            for index, row in class_medians.iterrows():
                class_median = row.filter(regex='^median_emb_').values
                distance = np.linalg.norm(embedding - class_median)
                if distance < distance_threshold * row['median_distance']:
                    matches.append((index, *location, distance))
        return matches

    args = get_args()
    embedding_directory = args.embedding_directory
    median_embeddings_path = args.median_embeddings_path
    matches_output_path = args.matches_output_path
    distance_threshold = args.distance_threshold
    
    embedding_dataset = load_embeddings(embedding_directory)
    median_embeddings_df = pd.read_csv(median_embeddings_path)
    
    all_locations = [(x, y, z) for z in range(embedding_dataset.shape[3]) for y in range(embedding_dataset.shape[2]) for x in range(embedding_dataset.shape[1])]
    batch_size = 1000
    batches = [all_locations[i:i + batch_size] for i in range(0, len(all_locations), batch_size)]

    matches = []
    with ProcessPoolExecutor() as executor:
        for batch in batches:
            current_embeddings = executor.submit(median_embedding, embedding_dataset, batch).result()
            matches.extend(match_embeddings(median_embeddings_df, current_embeddings, distance_threshold))
    
    matches_df = pd.DataFrame(matches, columns=['class', 'x', 'y', 'z', 'distance'])
    matches_df.to_csv(matches_output_path, index=False)
    print(f"Matches saved to {matches_output_path}")

setup(
    group="copick",
    name="discover-picks",
    version="0.0.4",
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
