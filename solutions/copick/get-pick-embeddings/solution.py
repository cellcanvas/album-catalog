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
    import json
    import zarr
    import numpy as np
    import pandas as pd

    def extract_picks(pick_file):
        """
        Extract pick information from the JSON file.
        """
        with open(pick_file, 'r') as file:
            data = json.load(file)
        return data

    def load_embeddings(zarr_directory):
        """
        Load the Zarr dataset containing embeddings.
        """
        return zarr.open(zarr_directory, mode='r')

    def fetch_embedding(embedding_dataset, location, scale_factor=10):
        # Adjust coordinates to match the Zarr scale and shift them to the correct dimensions
        spatial_coords = np.round(np.array([location['x'], location['y'], location['z']]) / scale_factor).astype(int)
        # Ensure spatial coordinates do not exceed dataset spatial dimensions
        spatial_coords = np.clip(spatial_coords, [0, 0, 0], np.array(embedding_dataset.shape[1:4]) - 1)
        # Access the embedding for the entire first dimension and specified spatial coordinates
        full_embedding = embedding_dataset[:, spatial_coords[0], spatial_coords[1], spatial_coords[2]]
        # Extract the direct embedding
        direct_embedding = full_embedding
        # Calculate median embedding within a radius around the spatial coordinates
        x_range = slice(max(0, spatial_coords[0]-3), min(embedding_dataset.shape[1], spatial_coords[0]+4))
        y_range = slice(max(0, spatial_coords[1]-3), min(embedding_dataset.shape[2], spatial_coords[1]+4))
        z_range = slice(max(0, spatial_coords[2]-3), min(embedding_dataset.shape[3], spatial_coords[2]+4))
        # Fetch the embeddings within the specified cube and calculate their median
        median_embedding = np.median(embedding_dataset[:, x_range, y_range, z_range], axis=(1, 2, 3))
        return direct_embedding, median_embedding

    def process_picks(picks_directory, embedding_dataset):
        """
        Process picks by creating a DataFrame of embeddings.
        """
        data = []
        for pick_file in os.listdir(picks_directory):
            full_path = os.path.join(picks_directory, pick_file)
            print(f"Processing pick file: {full_path}")
            pick_data = extract_picks(full_path)
            for point in pick_data['points']:
                direct_emb, median_emb = fetch_embedding(embedding_dataset, point['location'])
                embedding_info = {}
                for i, emb in enumerate(direct_emb):
                    embedding_info[f'direct_emb_{i}'] = emb
                for i, emb in enumerate(median_emb):
                    embedding_info[f'median_emb_{i}'] = emb
                data.append({
                    'run_name': pick_data['run_name'],
                    'user_id': pick_data['user_id'],
                    'prepick': pick_data['session_id'],
                    'pickable_object_name': pick_data['pickable_object_name'],
                    'location_x': point['location']['x'],
                    'location_y': point['location']['y'],
                    'location_z': point['location']['z'],
                    **embedding_info
                })
        return pd.DataFrame(data)

    args = get_args()
    picks_directory = args.picks_directory
    embedding_directory = args.embedding_directory
    dataframe_path = args.dataframe_path
    
    embedding_dataset = load_embeddings(embedding_directory)
    picks_df = process_picks(picks_directory, embedding_dataset)

    # Save the DataFrame to the specified path
    picks_df.to_csv(dataframe_path, index=False)
    print(f"Data saved to {dataframe_path}")

setup(
    group="copick",
    name="get-pick-embeddings",
    version="0.0.5",
    title="Analyze Picks and Corresponding Embeddings for a Single Run",
    description="Generates a DataFrame from picks and their corresponding embeddings for a single run and saves it.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "picks", "embedding", "cryoet"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "embedding_directory", "type": "string", "required": True, "description": "Path to the embedding Zarr directory."},
        {"name": "picks_directory", "type": "string", "required": True, "description": "Directory containing all the pick files."},
        {"name": "dataframe_path", "type": "string", "required": True, "description": "Path for the output dataframe."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
