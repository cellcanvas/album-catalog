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
    from copick.impl.filesystem import CopickRootFSSpec
    from numcodecs import Blosc

    def fetch_embeddings(tomo, feature_types, location, scale_factor=10):
        results = {}
        spatial_coords = np.round(np.array([location['x'], location['y'], location['z']]) / scale_factor).astype(int)
        spatial_coords = np.clip(spatial_coords, [0, 0, 0], np.array(tomo.shape[1:4]) - 1)

        for feature_type in feature_types:
            feature_data = tomo.get_feature_data(feature_type)
            if feature_data is not None:
                dataset = zarr.open(feature_data, mode='r')
                full_embedding = dataset[:, spatial_coords[0], spatial_coords[1], spatial_coords[2]]
                x_range = slice(max(0, spatial_coords[0]-3), min(dataset.shape[1], spatial_coords[0]+4))
                y_range = slice(max(0, spatial_coords[1]-3), min(dataset.shape[2], spatial_coords[1]+4))
                z_range = slice(max(0, spatial_coords[2]-3), min(dataset.shape[3], spatial_coords[2]+4))
                median_embedding = np.median(dataset[:, x_range, y_range, z_range], axis=(1, 2, 3))
                results[feature_type] = (full_embedding, median_embedding)
            else:
                results[feature_type] = (None, None)
        return results

    def process_run(run, feature_types):
        data = []
        picks = run.get_picks()
        tomo = run.get_tomogram()

        for pick in picks:
            embeddings = fetch_embeddings(tomo, feature_types, pick['location'])
            embedding_info = {}
            for feature_type, (direct_emb, median_emb) in embeddings.items():
                if direct_emb is not None:
                    for j, emb in enumerate(direct_emb):
                        embedding_info[f'{feature_type}_direct_{j}'] = emb
                    for j, emb in enumerate(median_emb):
                        embedding_info[f'{feature_type}_median_{j}'] = emb
            data.append({
                'run_name': run.name,
                'user_id': pick['user_id'],
                'session_id': pick['session_id'],
                'object_type': pick['object_type'],
                'location_x': pick['location']['x'],
                'location_y': pick['location']['y'],
                'location_z': pick['location']['z'],
                **embedding_info
            })
        return pd.DataFrame(data)

    args = get_args()
    run_name = args.run_name
    feature_types = args.feature_types.split(',')
    dataframe_path = args.dataframe_path
    copick_config_path = args.copick_config_path


    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")
    
    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")

    picks_df = process_run(run, feature_types)

    picks_df.to_csv(dataframe_path, index=False)
    print(f"Data saved to {dataframe_path}")

setup(
    group="copick",
    name="get-pick-embeddings",
    version="0.2.0",
    title="Analyze Picks and Corresponding Embeddings for a Single Run",
    description="Generates a DataFrame from picks and their corresponding embeddings for a single run and saves it.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "picks", "embedding", "cryoet"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the run to process."},
        {"name": "feature_types", "type": "string", "required": True, "description": "Comma-separated list of feature types to extract embeddings for."},
        {"name": "dataframe_path", "type": "string", "required": True, "description": "Path for the output dataframe."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
