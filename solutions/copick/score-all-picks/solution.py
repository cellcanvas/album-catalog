###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - scipy
  - scikit-image
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pandas
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import numpy as np
    import zarr
    import pandas as pd
    import json
    from sklearn.metrics import precision_recall_fscore_support
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    def compute_binary_stats(df):
        # Binary classification based on correct pick
        correct = df['pick_label'] == df['segmentation_label']
        binary_labels = [1 if label else 0 for label in correct]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            [1] * len(correct), binary_labels, average='binary', zero_division=0
        )

        return precision, recall, f1, len(df)

    def compute_pair_stats(df, object_id_to_name):
        if df.empty:
            return pd.Series({
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan,
                'num_picks': 0,
                **{f'num_{object_id_to_name[label]}': 0 for label in object_id_to_name}
            })
        
        precision, recall, f1, num_picks = compute_binary_stats(df)
        
        # Calculate the count of picks for each segmentation label
        num_class_counts = df['segmentation_label'].value_counts().to_dict()
        stats = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_picks': num_picks,
            **{f'num_{object_id_to_name[label]}': count for label, count in num_class_counts.items() if label in object_id_to_name}
        }
        
        # Ensure all expected keys are in the stats
        for label in object_id_to_name:
            key = f'num_{object_id_to_name[label]}'
            if key not in stats:
                stats[key] = 0

        return pd.Series(stats)

    def load_multilabel_segmentation(run, segmentation_name, voxel_spacing):
        print(f"Loading segmentation for run {run} with name {segmentation_name}")
        segs = run.get_segmentations(is_multilabel=True, name=segmentation_name, voxel_size=voxel_spacing)
        if not segs:
            print("No segmentation found.")
            return None
        seg = segs[0]
        group = zarr.open_group(seg.path, mode='r')
        if 'data' not in group:
            print("No 'data' dataset found in segmentation.")
            return None
        return group['data'][:]  # Use [:] to load the entire array into memory

    def validate_voxel_coordinates(seg, voxel_point):
        z, y, x = voxel_point
        shape = seg.shape
        return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    voxel_spacing = int(args.voxel_spacing)
    segmentation_idx_offset = int(args.segmentation_idx_offset)
    output_stats_dir = args.output_stats_dir

    root = CopickRootFSSpec.from_file(copick_config_path)
    all_runs = root.runs
    
    # Create a dictionary to map object names to IDs and reverse map
    object_name_to_id = {obj.name: obj.label for obj in root.config.pickable_objects}
    object_id_to_name = {label: name for name, label in object_name_to_id.items()}
    
    all_pick_stats = []  # Store all picks data for aggregate stats
    all_pair_stats = []  # Store per-run statistics
    
    for run in all_runs:
        run_name = run.meta.name
        
        print(f"Processing run: {run_name}")

        # Load segmentation
        seg = load_multilabel_segmentation(run, painting_segmentation_name, voxel_spacing) + segmentation_idx_offset

        if seg is None:
            print(f"Skipping run {run_name} as no segmentation found.")
            continue
        
        # Load all picks and group by user_id, session_id, and object_name
        pick_stats = []

        for obj in root.config.pickable_objects:
            for pick_set in run.get_picks(obj.name):
                try:
                    if not pick_set or not pick_set.points:
                        continue
                    
                    user_id = pick_set.user_id
                    session_id = pick_set.session_id
                    print(f"Processing picks for user {user_id} in session {session_id} for object {obj.name}")

                    # Skip specific sessions
                    if session_id in ["23982", "cellcanvasCandidates001", "cellcanvasCandidates002", "cellcanvasCandidates003", "cellcanvasCandidates004", "cellcanvasCandidates005", "cellcanvasCandidates006", "session1"]:
                        print("Skipping session")
                        continue
                    
                    for pick in pick_set.points:
                        point = pick.location
                        voxel_point = (int(point.z / voxel_spacing), int(point.y / voxel_spacing), int(point.x / voxel_spacing))

                        if not validate_voxel_coordinates(seg, voxel_point):
                            print(f"Skipping pick at {voxel_point} for user {user_id} in session {session_id} - out of bounds.")
                            continue                    
                        
                        label_in_seg = seg[voxel_point]
                        
                        pick_stats.append({
                            "run_name": run_name,
                            "user_id": user_id,
                            "session_id": session_id,
                            "object_name": obj.name,
                            "object_id": object_name_to_id[obj.name],
                            "pick_label": object_name_to_id[obj.name],
                            "segmentation_label": label_in_seg
                        })
                except json.JSONDecodeError as e:
                    print(f"Skipping pick set for {obj.name} due to JSON decode error: {e}")
                    continue

        # Create DataFrame
        pick_stats_df = pd.DataFrame(pick_stats)
        all_pick_stats.append(pick_stats_df)  # Append to overall picks list

        if not pick_stats_df.empty:
            # Aggregate statistics per (user_id, session_id, object_name) for this run
            pair_stats = pick_stats_df.groupby(['user_id', 'session_id', 'object_name']).apply(lambda df: compute_pair_stats(df, object_id_to_name)).reset_index()
            pair_stats['run_name'] = run_name
            all_pair_stats.append(pair_stats)

            # Save per-run statistics with tabs as delimiters
            pair_stats.to_csv(os.path.join(output_stats_dir, f"{run_name}_pick_stats.tsv"), sep='\t', index=False)

    # Combine all runs' statistics for aggregate analysis
    if all_pick_stats:
        combined_pick_stats_df = pd.concat(all_pick_stats)
        aggregate_stats = combined_pick_stats_df.groupby(['user_id', 'session_id', 'object_name']).apply(lambda df: compute_pair_stats(df, object_id_to_name)).reset_index()
        aggregate_stats.to_csv(os.path.join(output_stats_dir, "aggregate_pair_stats.tsv"), sep='\t', index=False)

    print("Statistics collection and saving complete.")

setup(
    group="copick",
    name="score-all-picks",
    version="0.0.13",
    title="Evaluate Picks Against Multilabel Segmentation",
    description="A solution that evaluates picks from a Copick project against a multilabel segmentation and computes metrics for each (user_id, session_id, object_name) pair for each run and across all runs.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "segmentation", "evaluation", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_name", "type": "string", "required": True, "description": "Name of the painting segmentation."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "segmentation_idx_offset", "type": "integer", "required": False, "description": "Offset for segmentation indices (default 0).", "default": 0},
        {"name": "output_stats_dir", "type": "string", "required": True, "description": "Directory to save output statistics."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
