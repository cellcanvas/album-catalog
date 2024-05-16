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
  - pandas
  - scikit-learn==1.3.2
  - joblib
  - pip:
    - album
    - copick
"""

def run():
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import zarr
    from sklearn.utils.class_weight import compute_class_weight
    from copick.impl.filesystem import CopickRootFSSpec
    from typing import Protocol
    import time
    import concurrent.futures

    class SegmentationModel(Protocol):
        """Protocol for semantic segmentations models that are compatible with the SemanticSegmentationManager."""
        def fit(self, X, y): ...
        def predict(self, X): ...

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = int(args.voxel_spacing)
    output_dataframe_path = args.output_dataframe_path

    # Load the Copick root from the configuration file
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "paintingsegmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run):
        segs = run.get_segmentations(
            user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing
        )
        if len(segs) == 0:
            print(f"Segmentation does not exist seg name {painting_segmentation_name}, user id {user_id}, session id {session_id}")
            return None
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def get_embedding_zarr(run):
        """Retrieve the denoised tomogram embeddings."""
        return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"]

    def process_run(painting_seg, features):        
        if not painting_seg:
            print("Painting segmentation failed, skipping.")
            return None, None

        # embedding_zarr = get_embedding_zarr(run)

        if len(features) == 0:
            print("Missing features.")
            return None, None

        features = np.array(features)
        labels = np.array(painting_seg)

        # Flatten labels for boolean indexing
        flattened_labels = labels.flatten()

        # Compute valid_indices based on labels > 0
        valid_indices = np.nonzero(flattened_labels > 0)[0]

        # Flatten only the spatial dimensions of the dataset_features while preserving the feature dimension
        c, h, w, d = features.shape
        reshaped_features = features.reshape(c, h * w * d)

        # Apply valid_indices for each feature dimension separately
        filtered_features_list = [np.take(reshaped_features[i, :], valid_indices, axis=0) for i in range(c)]
        filtered_features = np.stack(filtered_features_list, axis=1)

        # Adjust labels
        filtered_labels = flattened_labels[valid_indices] - 1

        print(f"Processed run with {filtered_labels.shape[0]} valid samples")
        return filtered_features, filtered_labels

    # Function to load features and labels from Copick runs in parallel
    def load_features_and_labels_from_copick(root):
        all_features = []
        all_labels = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for run in root.runs:
                painting_seg = get_painting_segmentation(run)
                if painting_seg is None:
                    print(f"Missing painting seg: {run}")
                else:
                    print(f"Not missing in {run}")
                tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised")
                if not tomo:
                    # Skip if tomogram doesnt exist                                                                                                                                                         
                    continue
                features = tomo.features
                if len(features) > 0:  
                    features = zarr.open(features[0].path, "r")
                    futures.append(executor.submit(process_run, painting_seg, features))
                else:
                    print(f"Job not submitted for {run}")

            for future in concurrent.futures.as_completed(futures):
                print("a run finished!")
                filtered_features, filtered_labels = future.result()
                if filtered_features is not None and filtered_labels is not None:
                    all_features.append(filtered_features)
                    all_labels.append(filtered_labels)

        if len(all_features) > 0 and len(all_labels) > 0:
            all_features = np.concatenate(all_features)
            all_labels = np.concatenate(all_labels)
        return all_features, all_labels

    # Extract training data from Copick runs
    print("Extracting data from Copick runs...")
    features_all, labels_all = load_features_and_labels_from_copick(root)

    if features_all.size > 0 and labels_all.size > 0:
        print(f"Total samples: {features_all.shape[0]}, Total features per sample: {features_all.shape[1]}")
    else:
        print("No features.")
        return
    
    if labels_all.size == 0:
        print("No labels present. Skipping model update.")
    else:
        # Save features and labels into a DataFrame
        print("Saving features and labels to DataFrame")
        df = pd.DataFrame(features_all)
        df['label'] = labels_all
        df.to_csv(output_dataframe_path, index=False)
        print(f"Features and labels saved to {output_dataframe_path}")

setup(
    group="copick",
    name="labeled-data-from-picks",
    version="0.0.1",
    title="Process Copick Runs and Save Features and Labels",
    description="A solution that processes all Copick runs and saves the resulting features and labels into a DataFrame.",
    solution_creators=["Kyle Harrington"],
    tags=["copick", "features", "labels", "dataframe"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_name", "type": "string", "required": False, "description": "Name for the painting segmentation."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "output_dataframe_path", "type": "string", "required": True, "description": "Path for the output CSV file containing the features and labels."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
