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
  - py-xgboost-gpu
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
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    input_zarr_path = args.input_zarr_path
    output_model_path = args.output_model_path

    # Default parameters
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    learning_rate = args.learning_rate
    class_weights_str = args.class_weights

    def parse_class_weights(class_weights_str, unique_labels):
        """Parse class weights from a comma-separated string and pad with 0 for missing weights."""
        class_weights_list = list(map(float, class_weights_str.split(',')))
        class_weights_dict = {label: (class_weights_list[i] if i < len(class_weights_list) else 0.0) for i, label in enumerate(unique_labels)}
        return class_weights_dict

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the XGBoost model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    # Function to find the minimum feature size by checking arrays in the Zarr
    def find_min_feature_size(zarr_path):
        zarr_store = zarr.open(ZipStore(zarr_path, mode='r'), mode='r')
        max_size = 0

        for run_key in zarr_store.keys():
            run_group = zarr_store[run_key]
            features = run_group['features'][:]
            if features.shape[1] > max_size:
                max_size = features.shape[1]

        return max_size

    # Function to load data from Zarr store
    def load_data_from_zarr(zarr_path, min_feature_size):
        zarr_store = zarr.open(ZipStore(zarr_path, mode='r'), mode='r')
        features_list = []
        labels_list = []

        for run_key in zarr_store.keys():
            run_group = zarr_store[run_key]
            features = run_group['features'][:]
            labels = run_group['labels'][:]

            if features.shape[1] >= min_feature_size and len(np.unique(labels)) > 1:  # Exclude small arrays and runs with only one label
                features_list.append(features)
                labels_list.append(labels)

        all_features = np.concatenate(features_list)
        all_labels = np.concatenate(labels_list)
        return all_features, all_labels

    # Determine min_feature_size
    min_feature_size = find_min_feature_size(input_zarr_path)
    logger.info(f"Determined minimum feature size: {min_feature_size}")

    # Load features and labels
    logger.info(f"Loading data from {input_zarr_path}")
    features, labels = load_data_from_zarr(input_zarr_path, min_feature_size)

    if features.size > 0 and labels.size > 0:
        logger.info(f"Total samples: {features.shape[0]}, Total features per sample: {features.shape[1]}")
    else:
        logger.error("No features or labels found.")
        return

    # Encode labels to contiguous integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Calculate or parse class weights
    unique_labels = np.unique(encoded_labels)
    if class_weights_str:
        class_weights = parse_class_weights(class_weights_str, unique_labels)
    else:
        class_weights = calculate_class_weights(encoded_labels)
    logger.info(f"Class weights: {class_weights}")

    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[label] for label in encoded_labels])

    # Train XGBoost with 10-fold cross-validation
    logger.info(f"Training XGBoost with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, and 10-fold cross-validation...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softmax',
        tree_method='hist',
        device='cuda',  # Use CUDA device
        verbosity=2,
        eval_metric='mlogloss'
    )
    skf = StratifiedKFold(n_splits=10)
    
    scores = []
    for train_index, test_index in skf.split(features, encoded_labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights[train_index])
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model.fit(X_train, y_train, sample_weight=sample_weights[train_index])
        score = model.score(X_test, y_test)
        scores.append(score)
        
    scores = np.array(scores)
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean accuracy: {scores.mean()}, Std: {scores.std()}")

    # Train the final model on all data
    logger.info("Training final model on all data...")
    dtrain = xgb.DMatrix(features, label=encoded_labels, weight=sample_weights)
    model.fit(features, encoded_labels, sample_weight=sample_weights)

    # Save the trained model and label encoder
    logger.info(f"Saving model and label encoder to {output_model_path}")
    joblib.dump((model, label_encoder), output_model_path)
    logger.info("Model and label encoder saved successfully")

    logger.info(f"XGBoost model trained and saved to {output_model_path}")

setup(
    group="cellcanvas",
    name="train-model-xgboost",
    version="0.0.4",
    title="Train XGBoost on Zarr Data with Cross-Validation",
    description="A solution that trains an XGBoost model using data from a Zarr zip store, filters runs with only one label, and performs 10-fold cross-validation.",
    solution_creators=["Your Name"],
    tags=["xgboost", "machine learning", "segmentation", "training", "cross-validation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "input_zarr_path", "type": "string", "required": True, "description": "Path to the input Zarr zip store containing the features and labels."},
        {"name": "output_model_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained XGBoost model."},
        {"name": "n_estimators", "type": "integer", "required": False, "description": "Number of trees in the XGBoost model.", "default": 750},
        {"name": "max_depth", "type": "integer", "required": False, "description": "The maximum depth of the trees.", "default": 18},
        {"name": "learning_rate", "type": "float", "required": False, "description": "The learning rate.", "default": 0.1},
        {"name": "class_weights", "type": "string", "required": False, "description": "Class weights for the XGBoost model as a comma-separated list.", "default": ""}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
