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
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import json
    import zarr
    import numpy as np
    import pandas as pd
    from copick.impl.filesystem import CopickRootFSSpec
    from numcodecs import Blosc

    def fetch_median_embedding(tomo, feature_types, location, radius):
        results = {}
        spatial_coords = np.round(np.array([location['x'], location['y'], location['z']])).astype(int)
        spatial_coords = np.clip(spatial_coords, [0, 0, 0], np.array(tomo.shape[1:4]) - 1)

        for feature_type in feature_types:
            feature_data = tomo.get_features(feature_type)
            if feature_data is not None:
                dataset = zarr.open(feature_data.zarr(), mode='r')
                x_range = slice(max(0, spatial_coords[0]-radius), min(dataset.shape[1], spatial_coords[0]+radius+1))
                y_range = slice(max(0, spatial_coords[1]-radius), min(dataset.shape[2], spatial_coords[1]+radius+1))
                z_range = slice(max(0, spatial_coords[2]-radius), min(dataset.shape[3], spatial_coords[2]+radius+1))
                median_embedding = np.median(dataset[:, x_range, y_range, z_range], axis=(1, 2, 3))
                results[feature_type] = median_embedding
            else:
                results[feature_type] = None
        return results

    def process_run(run, feature_types, radius, user_ids):
        data = {}
        picks = run.get_picks()
        tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)

        for pick in picks:
            if user_ids and pick['user_id'] not in user_ids:
                continue
            object_type = pick['object_type']
            if object_type not in data:
                data[object_type] = []
            embeddings = fetch_median_embedding(tomo, feature_types, pick['location'], radius)
            embedding_info = {}
            for feature_type, median_emb in embeddings.items():
                if median_emb is not None:
                    for j, emb in enumerate(median_emb):
                        embedding_info[f'{feature_type}_median_{j}'] = emb
            data[object_type].append(embedding_info)
        
        median_embeddings = {obj_type: np.median([list(emb.values()) for emb in embs], axis=0).tolist()
                             for obj_type, embs in data.items()}
        return median_embeddings

    args = get_args()
    run_names = args.run_names.split(',')
    feature_types = args.feature_types.split(',')
    radius = int(args.radius)
    output_path = args.output_path
    tomo_type = args.tomo_type
    voxel_spacing = int(args.voxel_spacing)    
    copick_config_path = args.copick_config_path
    user_ids = args.user_ids.split(',') if args.user_ids else []

    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    combined_median_embeddings = {}
    
    for run_name in run_names:
        run = root.get_run(run_name)
        if run is None:
            print(f"Run with name '{run_name}' not found. Skipping...")
            continue

        median_embeddings = process_run(run, feature_types, radius, user_ids)
        for obj_type, embeddings in median_embeddings.items():
            if obj_type in combined_median_embeddings:
                combined_median_embeddings[obj_type].append(embeddings)
            else:
                combined_median_embeddings[obj_type] = [embeddings]

    final_median_embeddings = {obj_type: np.median(embs, axis=0).tolist()
                               for obj_type, embs in combined_median_embeddings.items()}

    with open(output_path, 'w') as f:
        json.dump(final_median_embeddings, f, indent=4)
    print(f"Median embeddings saved to {output_path}")

setup(
    group="copick",
    name="get-median-embeddings",
    version="0.0.2",
    title="Analyze Median Embeddings for Each Object Type Across Multiple Runs",
    description="Generates a file containing the median embeddings for each object type based on the picks in multiple runs, filtered by user IDs if provided.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "picks", "embedding", "cryoet"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "run_names", "type": "string", "required": True, "description": "Comma-separated list of run names to process."},
        {"name": "feature_types", "type": "string", "required": True, "description": "Comma-separated list of feature types to extract embeddings for."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram, e.g. denoised."},        
        {"name": "radius", "type": "integer", "required": True, "description": "Radius for calculating median embeddings."},
        {"name": "user_ids", "type": "string", "required": False, "description": "Comma-separated list of user IDs to filter picks by. If not provided, all picks will be processed."},
        {"name": "output_path", "type": "string", "required": True, "description": "Path for the output file."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
