###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
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
  - optuna
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
    from zipfile import BadZipFile
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import xgboost as xgb
    import logging
    import optuna

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    input_zarr_path = args.input_zarr_path
    output_model_path = args.output_model_path
    best_params_path = args.best_params_path
    num_trials = args.num_trials
    n_splits = int(args.n_splits)
    subset_size = int(args.subset_size)
    seed = int(args.seed)
    objective_function = args.objective_function

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the XGBoost model."""
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        return {label: weight for label, weight in zip(np.unique(labels), class_weights)}

    def load_and_balance_data(zarr_path, subset_size, seed):
        features, labels = load_data(zarr_path)

        if features is None or labels is None:
            return None, None

        return create_balanced_subset(features, labels, subset_size)  # Balance the entire dataset at once

    def load_data(zarr_path):
        try:
            zarr_store = zarr.open(ZipStore(zarr_path, mode='r'), mode='r')
        except BadZipFile:
            logger.error(f"Bad zip file: {zarr_path}. Skipping...")
            return None, None

        features_list = []
        labels_list = []

        expected_feature_size = None

        for run_key in zarr_store.keys():
            run_group = zarr_store[run_key]
            features = run_group['features'][:]
            labels = run_group['labels'][:]

            if expected_feature_size is None:
                expected_feature_size = features.shape[1]
                print(f"Expected feature size: {expected_feature_size}")

            if features.shape[1] == expected_feature_size and len(np.unique(labels)) > 1:  # Exclude small arrays and runs with only one label
                features_list.append(features)
                labels_list.append(labels)
            else:
                logger.warning(f"Skipping run {run_key} due to unexpected feature size or single label: feature size {features.shape[1]}, num labels {len(np.unique(labels))}")

        if features_list and labels_list:
            all_features = np.concatenate(features_list)
            all_labels = np.concatenate(labels_list)
            return all_features, all_labels
        else:
            return None, None

    def create_balanced_subset(features, labels, subset_size):
        """Create a balanced subset of the data."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_class_size = min(counts)

        if min_class_size == 0:
            raise ValueError("At least one class has no samples")

        # Ensure we don't exceed the requested subset_size
        min_class_size = min(min_class_size, subset_size // len(unique_labels))

        balanced_features = []
        balanced_labels = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(label_indices, min_class_size, replace=False)
            balanced_features.append(features[sampled_indices])
            balanced_labels.append(labels[sampled_indices])

        return np.concatenate(balanced_features), np.concatenate(balanced_labels)

    def get_scorers():
        """Return a dictionary of scoring functions."""
        return {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }

    def objective(trial, balanced_features, encoded_labels, sample_weights):
        params = {
            'eta': trial.suggest_float('eta', 0.01, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10.0),
            'max_depth': trial.suggest_int('max_depth', 0, 30),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 10.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 10.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 0, 10.0),
            'alpha': trial.suggest_float('alpha', 0, 10.0),
            'max_bin': trial.suggest_int('max_bin', 32, 512),
            'objective': 'multi:softmax',
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'num_class': len(np.unique(encoded_labels))  # Specify the number of classes
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scorers = get_scorers()
        results = {name: [] for name in scorers.keys()}

        for train_index, test_index in skf.split(balanced_features, encoded_labels):
            X_train, X_test = balanced_features[train_index], balanced_features[test_index]
            y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights[train_index])
            dtest = xgb.DMatrix(X_test, label=y_test)

            bst = xgb.train(params, dtrain, evals=[(dtest, 'eval')], verbose_eval=False)
            preds = bst.predict(dtest)
            for name, scorer in scorers.items():
                if name in ['f1', 'precision', 'recall']:
                    score = scorer(y_test, preds, average='weighted')
                else:
                    score = scorer(y_test, preds)
                results[name].append(score)

        mean_scores = {name: np.mean(scores) for name, scores in results.items()}
        for name, mean_score in mean_scores.items():
            logger.info(f"Trial {trial.number}: Mean {name} = {mean_score:.4f}")

        return mean_scores[objective_function]

    def main():
        features, labels = load_and_balance_data(input_zarr_path, subset_size, seed)

        if features is None or labels is None:
            return  # Exit if data loading failed

        # Encode labels to contiguous integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Calculate class weights
        class_weights = calculate_class_weights(encoded_labels)

        # Convert class weights to sample weights
        sample_weights = np.array([class_weights[label] for label in encoded_labels])

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, features, encoded_labels, sample_weights), n_trials=num_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.params}")

        # Save the best parameters
        with open(best_params_path, 'w') as f:
            f.write(f"Best trial parameters:\n")
            f.write(f"Params: {study.best_trial.params}\n")
            f.write(f"Value: {study.best_trial.value}\n")
            f.write(f"Trial: {study.best_trial.number}\n")

        # Save the best model
        best_params = study.best_trial.params

        params = {
            'eta': best_params['eta'],
            'gamma': best_params['gamma'],
            'max_depth': best_params['max_depth'],
            'min_child_weight': best_params['min_child_weight'],
            'max_delta_step': best_params['max_delta_step'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'lambda': best_params['lambda'],
            'alpha': best_params['alpha'],
            'max_bin': best_params['max_bin'],
            'objective': 'multi:softmax',
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'num_class': len(np.unique(encoded_labels))  # Specify the number of classes
        }

        dtrain = xgb.DMatrix(features, label=encoded_labels, weight=sample_weights)
        final_model = xgb.train(params, dtrain)

        # Save the trained model and label encoder
        logger.info(f"Saving model and label encoder to {output_model_path}")
        joblib.dump((final_model, label_encoder), output_model_path)
        logger.info("Model and label encoder saved successfully")

    main()

setup(
    group="cellcanvas",
    name="optimize-xgboost",
    version="0.0.6",
    title="Optimize XGBoost with Optuna on Zarr Data",
    description="A solution that optimizes an XGBoost model using Optuna, data from a Zarr zip store, and performs 10-fold cross-validation.",
    solution_creators=["Kyle Harrington"],
    tags=["xgboost", "machine learning", "segmentation", "training", "cross-validation", "optuna"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "input_zarr_path", "type": "string", "required": True, "description": "Path to the input Zarr zip store containing the features and labels."},
        {"name": "output_model_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained XGBoost model."},
        {"name": "best_params_path", "type": "string", "required": True, "description": "Path for the output file containing the best parameters from Optuna."},
        {"name": "n_splits", "type": "string", "required": True, "description": "Number of splits for cross-validation."},
        {"name": "subset_size", "type": "string", "required": True, "description": "Total number of points for balanced subset."},
        {"name": "seed", "type": "string", "required": True, "description": "Random seed for reproducibility."},
        {"name": "num_trials", "type": "integer", "required": True, "description": "Number of Optuna trials to run."},
        {"name": "objective_function", "type": "string", "required": True, "description": "Objective function to optimize. Options are: accuracy, f1, precision, recall."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
