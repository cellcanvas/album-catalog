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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    input_zarr_path = args.input_zarr_path
    output_model_path = args.output_model_path
    n_estimators = int(args.n_estimators)

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))
    
    # Function to load data from Zarr store
    def load_data_from_zarr(zarr_path):
        zarr_store = zarr.open(ZipStore(zarr_path, mode='r'), mode='r')
        features_list = []
        labels_list = []

        for run_key in zarr_store.keys():
            run_group = zarr_store[run_key]
            features = run_group['features'][:]
            labels = run_group['labels'][:]

            if len(np.unique(labels)) > 1:  # Exclude runs with only one label
                features_list.append(features)
                labels_list.append(labels)

        all_features = np.concatenate(features_list)
        all_labels = np.concatenate(labels_list)
        return all_features, all_labels

    # Load features and labels
    logger.info(f"Loading data from {input_zarr_path}")
    features, labels = load_data_from_zarr(input_zarr_path)

    if features.size > 0 and labels.size > 0:
        logger.info(f"Total samples: {features.shape[0]}, Total features per sample: {features.shape[1]}")
    else:
        logger.error("No features or labels found.")
        return

    

    # Calculate class weights
    class_weights = calculate_class_weights(labels)
    logger.info(f"Class weights calculated: {class_weights}")

    # Train Random Forest with 10-fold cross-validation
    logger.info(f"Training Random Forest with {n_estimators} estimators and 10-fold cross-validation...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        max_depth=15,
        max_samples=0.05,
        class_weight=class_weights
    )
    skf = StratifiedKFold(n_splits=10)
    scores = cross_val_score(model, features, labels, cv=skf, scoring='accuracy')
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean accuracy: {scores.mean()}, Std: {scores.std()}")

    # Train the final model on all data
    logger.info("Training final model on all data...")
    model.fit(features, labels)

    # Save the trained model
    logger.info(f"Saving model to {output_model_path}")
    joblib.dump(model, output_model_path)
    logger.info("Model saved successfully")

    logger.info(f"Random Forest model trained and saved to {output_model_path}")

setup(
    group="cellcanvas",
    name="train-model",
    version="0.1.0",
    title="Train Random Forest on Zarr Data with Cross-Validation",
    description="A solution that trains a Random Forest model using data from a Zarr zip store, filters runs with only one label, and performs 10-fold cross-validation.",
    solution_creators=["Kyle Harrington"],
    tags=["random forest", "machine learning", "segmentation", "training", "cross-validation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "input_zarr_path", "type": "string", "required": True, "description": "Path to the input Zarr zip store containing the features and labels."},
        {"name": "output_model_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained Random Forest model."},
        {"name": "n_estimators", "type": "string", "required": True, "description": "Number of trees in the Random Forest."},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
