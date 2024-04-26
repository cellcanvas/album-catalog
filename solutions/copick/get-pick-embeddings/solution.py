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
        """
        Fetch the direct embedding and a median embedding within a specified radius.
        """
        # Adjust coordinates to match the Zarr scale
        coords = np.round(np.array([location['x'], location['y'], location['z']]) / scale_factor).astype(int)
        direct_embedding = embedding_dataset[tuple(coords)]
        
        # Calculate median embedding within a radius
        x_range = slice(max(0, coords[0]-3), coords[0]+4)
        y_range = slice(max(0, coords[1]-3), coords[1]+4)
        z_range = slice(max(0, coords[2]-3), coords[2]+4)
        median_embedding = np.median(embedding_dataset[x_range, y_range, z_range], axis=(0, 1, 2))
        
        return direct_embedding, median_embedding

    def process_run(run_directory, picks_directory, embedding_subdirectory):
        """
        Process each run by creating a DataFrame of picks and embeddings.
        """
        data = []
        embedding_directory = os.path.join(run_directory, embedding_subdirectory)
        embedding_dataset = load_embeddings(embedding_directory)
        
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
    runs_directory = args.runs_directory
    embedding_subdirectory = args.embedding_subdirectory
    dataframe_path = args.dataframe_path
    
    all_runs_df = pd.DataFrame()
    for run_name in os.listdir(runs_directory):
        run_dir = os.path.join(runs_directory, run_name)
        picks_dir = os.path.join(run_dir, 'Picks')
        voxel_dir = os.path.join(run_dir, embedding_subdirectory)
        
        if os.path.exists(picks_dir) and os.path.exists(voxel_dir):
            run_df = process_run(run_dir, picks_dir, voxel_dir)
            all_runs_df = pd.concat([all_runs_df, run_df], ignore_index=True)

    # Save the DataFrame to the specified path
    all_runs_df.to_csv(dataframe_path, index=False)
    print(f"Data saved to {dataframe_path}")

setup(
    group="copick",
    name="get-pick-embeddings",
    version="0.0.3",
    title="Analyze Picks and Corresponding Embeddings",
    description="Generates a DataFrame from picks and their corresponding embeddings and saves it.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "picks", "embedding", "cryoet"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "runs_directory", "type": "string", "required": True, "description": "Directory containing all the run directories."},
        {"name": "dataframe_path", "type": "string", "required": True, "description": "Path for the output dataframe."},
        {"name": "embedding_subdirectory", "type": "string", "required": True, "description": "Name of embeddings subdirectory."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
