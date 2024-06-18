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
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    class SegmentationModel(Protocol):
        """Protocol for semantic segmentations models that are compatible with the SemanticSegmentationManager."""
        def fit(self, X, y): ...
        def predict(self, X): ...

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_names = args.painting_segmentation_names.split(',')
    session_id = args.session_id
    user_id = args.user_id
    feature_types = args.feature_types.split(',')
    tomo_type = args.tomo_type
    voxel_spacing = float(args.voxel_spacing)
    output_zarr_path = args.output_zarr_path
    run_names = args.run_names.split(',') if args.run_names else None

    # Load the Copick root from the configuration file
    logger.info(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    logger.info("Copick root loaded successfully")

    def get_painting_segmentation(run, painting_name):
        try:
            segs = run.get_segmentations(
                user_id=user_id, is_multilabel=True, name=painting_name, voxel_size=voxel_spacing
            )
            if len(segs) == 0:
                # logger.info(f"Segmentation does not exist seg name {painting_name}, user id {user_id}, session id {session_id}")
                logger.info(f"Segmentation does not exist seg name {painting_name}, user id {user_id}")
                return None
            else:
                seg = segs[0]
                group = zarr.open_group(seg.path, mode="a")
                if 'data' not in group:
                    shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
                    group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
            return group['data']
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            logger.error(f"Error opening painting segmentation zarr: {e}")
            return None

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def get_embedding_zarr(run, feature_type):
        """Retrieve the tomotype tomogram embeddings."""
        try:
            return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"]
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            logger.error(f"Error opening embedding zarr for feature type {feature_type}: {e}")
            return None

    def process_run(run, painting_segmentation_names, voxel_spacing, user_id, session_id, zarr_store, feature_types):
        all_painting_segs = []
        for painting_segmentation_name in painting_segmentation_names:
            painting_seg = get_painting_segmentation(run, painting_segmentation_name)
            if painting_seg:
                all_painting_segs.append(painting_seg[:])
        
        if not all_painting_segs:
            logger.info(f"No valid painting segmentations found for run {run}, skipping.")
            return
        
        # Combine painting segmentations with rightmost array having the highest precedence
        combined_labels = np.zeros_like(all_painting_segs[0])
        for painting_seg in all_painting_segs:
            combined_labels = np.where(painting_seg > 0, painting_seg, combined_labels)
        
        tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        if not tomo:
            logger.info(f"No tomogram found for run {run}, skipping.")
            return

        all_features = []

        for feature_type in feature_types:
            features = tomo.features
            if len(features) == 0:
                logger.info(f"No features found for run {run}, skipping.")
                continue

            features_path = [f.path for f in features if f.feature_type == feature_type]
            if len(features_path) == 0:
                logger.info(f"No {feature_type} features found for run {run}, skipping.")
                continue
            
            try:            
                features = zarr.open(features_path[0], "r")[:]
                all_features.append(features)
            except (zarr.errors.PathNotFoundError, KeyError) as e:
                logger.error(f"Error opening features zarr for feature type {feature_type} in run {run}: {e}")
                continue

        if len(all_features) == 0:
            logger.info(f"No valid features found for run {run}, skipping.")
            return

        # Concatenate features along the feature dimension
        concatenated_features = np.concatenate(all_features, axis=0)

        labels = combined_labels

        if labels.size == 0:
            logger.info(f"No labels found for run {run}, skipping.")
            return

        # Flatten labels for boolean indexing
        flattened_labels = labels.reshape(-1)

        # Compute valid_indices based on labels > 0
        valid_indices = np.nonzero(flattened_labels > 0)[0]

        if valid_indices.size == 0:
            logger.info(f"No valid labels found for run {run}, skipping.")
            return

        # Flatten only the spatial dimensions of the concatenated_features while preserving the feature dimension
        c, h, w, d = concatenated_features.shape
        reshaped_features = concatenated_features.reshape(c, h * w * d)

        # Apply valid_indices for each feature dimension separately
        filtered_features_list = [np.take(reshaped_features[i, :], valid_indices, axis=0) for i in range(c)]
        filtered_features = np.stack(filtered_features_list, axis=1)

        # Adjust labels
        filtered_labels = flattened_labels[valid_indices] - 1

        logger.info(f"Processed run {run} with {filtered_labels.shape[0]} valid samples")

        # Create Zarr group for the run
        run_group = zarr_store.create_group(f"run_{run.name}")

        # Create subgroups for features and labels
        run_group.create_dataset('features', data=filtered_features, compressor=numcodecs.Blosc())
        run_group.create_dataset('labels', data=filtered_labels, compressor=numcodecs.Blosc())


    def load_features_and_labels_from_copick(root, output_zarr_path, run_names):
        zarr_store = zarr.open(ZipStore(output_zarr_path, mode='w'), mode='w')

        runs_to_process = [run for run in root.runs if run.name in run_names] if run_names else root.runs

        for run in runs_to_process:
            logger.info(f"Preparing run {run}")
            try:
                process_run(run, painting_segmentation_names, voxel_spacing, user_id, session_id, zarr_store, feature_types)
                logger.info("A run finished!")
            except Exception as e:
                logger.error(f"Error in processing run {run}: {e}")

    # Extract training data from Copick runs
    logger.info("Extracting data from Copick runs...")
    load_features_and_labels_from_copick(root, output_zarr_path, run_names)

    logger.info(f"Features and labels saved to {output_zarr_path}")

setup(
    group="copick",
    name="labeled-data-from-picks",
    version="0.1.6",
    title="Process Copick Runs and Save Features and Labels",
    description="A solution that processes all Copick runs and saves the resulting features and labels into a Zarr zip store.",
    solution_creators=["Kyle Harrington"],
    tags=["copick", "features", "labels", "dataframe"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_names", "type": "string", "required": True, "description": "Comma-separated list of names for the painting segmentations. Rightmost segmentation has highest precedence."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram, e.g. denoised."},
        {"name": "feature_types", "type": "string", "required": True, "description": "Comma-separated list of feature types to use for each tomogram, e.g. cellcanvas01,cellcanvas02."},
        {"name": "output_zarr_path", "type": "string", "required": True, "description": "Path for the output Zarr zip store containing the features and labels."},
        {"name": "run_names", "type": "string", "required": False, "description": "Comma-separated list of run names to process. If not provided, all runs will be processed."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
