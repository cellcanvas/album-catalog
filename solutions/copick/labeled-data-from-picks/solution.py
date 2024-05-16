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
  - h5py
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
    from zarr.storage import ZipStore
    from sklearn.utils.class_weight import compute_class_weight
    from copick.impl.filesystem import CopickRootFSSpec
    from typing import Protocol
    import time
    import concurrent.futures
    import numcodecs

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
    output_zarr_path = args.output_zarr_path

    # Load the Copick root from the configuration file
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "paintingsegmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run):
        try:
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
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            print(f"Error opening painting segmentation zarr: {e}")
            return None

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def get_embedding_zarr(run):
        """Retrieve the denoised tomogram embeddings."""
        try:
            return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"]
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            print(f"Error opening embedding zarr: {e}")
            return None

    def process_run(run, painting_segmentation_name, voxel_spacing, user_id, session_id, zarr_store):        
        painting_seg = get_painting_segmentation(run)
        if not painting_seg:
            print("Painting segmentation failed, skipping.")
            return

        tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised")
        if not tomo:
            return
        
        features = tomo.features
        if len(features) == 0:
            return
        
        try:
            features = zarr.open(features[0].path, "r")
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            print(f"Error opening features zarr: {e}")
            return

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

        # Create Zarr group for the run
        run_group = zarr_store.create_group(f"run_{run.id}")

        # Create subgroups for features and labels
        run_group.create_dataset('features', data=filtered_features, compressor=numcodecs.Blosc())
        run_group.create_dataset('labels', data=filtered_labels, compressor=numcodecs.Blosc())

    # Function to load features and labels from Copick runs in parallel
    def load_features_and_labels_from_copick(root, output_zarr_path):
        zarr_store = zarr.open(ZipStore(output_zarr_path, mode='w'), mode='w')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for run in root.runs:
                print(f"Preparing run {run}")
                futures.append(executor.submit(process_run, run, painting_segmentation_name, voxel_spacing, user_id, session_id, zarr_store))

            for future in concurrent.futures.as_completed(futures):
                print("A run finished!")

    # Extract training data from Copick runs
    print("Extracting data from Copick runs...")
    load_features_and_labels_from_copick(root, output_zarr_path)

    print(f"Features and labels saved to {output_zarr_path}")

setup(
    group="copick",
    name="labeled-data-from-picks",
    version="0.0.5",
    title="Process Copick Runs and Save Features and Labels",
    description="A solution that processes all Copick runs and saves the resulting features and labels into a Zarr zip store.",
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
        {"name": "output_zarr_path", "type": "string", "required": True, "description": "Path for the output Zarr zip store containing the features and labels."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
