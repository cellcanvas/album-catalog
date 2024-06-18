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
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    def compute_pair_stats(df):
        conf_matrix = confusion_matrix(df['segmentation_label'], df['pick_label'])
        precision, recall, f1, _ = precision_recall_fscore_support(df['segmentation_label'], df['pick_label'], average='weighted')
        return pd.Series({
            'confusion_matrix': json.dumps(conf_matrix.tolist()),  # Store matrix as JSON-encoded string
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_picks': len(df)
        })

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
        return group['data']
    
    def paint_picks_as_balls(painting_seg_array, voxel_location, segmentation_id, ball_radius):
        z, y, x = voxel_location
        # Implement the ball painting logic here if required
        painting_seg_array[z, y, x] = segmentation_id

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    voxel_spacing = int(args.voxel_spacing)
    segmentation_idx_offset = int(args.segmentation_idx_offset)
    output_stats_dir = args.output_stats_dir

    root = CopickRootFSSpec.from_file(copick_config_path)
    all_runs = root.runs
    
    # Create a dictionary to map object names to IDs
    object_name_to_id = {obj.name: obj.label for obj in root.config.pickable_objects}
    
    all_pair_stats = []
    
    for run in all_runs:
        run_name = run.meta.name
        
        print(f"Processing run: {run_name}")

        # Load segmentation
        seg = load_multilabel_segmentation(run, painting_segmentation_name, voxel_spacing)

        if seg is None:
            print(f"Skipping run {run_name} as no segmentation found.")
            continue
        
        # Load all picks and group by user_id, session_id, and object_name
        pick_stats = []

        for obj in root.config.pickable_objects:
            for pick_set in run.get_picks(obj.name):
                if not pick_set or not pick_set.points:
                    continue
                
                user_id = pick_set.user_id
                session_id = pick_set.session_id
                print(f"Processing picks for user {user_id} in session {session_id} for object {obj.name}")
                
                for pick in pick_set.points:
                    point = pick.location
                    voxel_point = (int(point['z'] / voxel_spacing), int(point['y'] / voxel_spacing), int(point['x'] / voxel_spacing))
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

        # Create DataFrame
        pick_stats_df = pd.DataFrame(pick_stats)

        if not pick_stats_df.empty:
            # Aggregate statistics per (user_id, session_id, object_name) for this run
            pair_stats = pick_stats_df.groupby(['user_id', 'session_id', 'object_name']).apply(compute_pair_stats)
            pair_stats['run_name'] = run_name
            all_pair_stats.append(pair_stats)

            # Save per-run statistics
            pair_stats.to_csv(os.path.join(output_stats_dir, f"{run_name}_pick_stats.csv"), index=False)

    # Combine all runs' statistics
    if all_pair_stats:
        aggregate_pair_stats = pd.concat(all_pair_stats)
        aggregate_pair_stats.to_csv(os.path.join(output_stats_dir, "aggregate_pair_stats.csv"), index=False)

    print("Statistics collection and saving complete.")

setup(
    group="copick",
    name="score-all-picks",
    version="0.0.6",
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
