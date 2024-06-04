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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
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

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def load_and_balance_data(zarr_path, subset_size, seed):
        features, labels = load_data(zarr_path)

        if features is None or labels is None:
            return None, None, None

        label_mapping = {i: label for i, label in enumerate(np.unique(labels))}

        balanced_datasets = {}
        for i, label in label_mapping.items():
            binary_labels = (labels == label).astype(int)
            balanced_features, balanced_binary_labels = create_balanced_subset(features, binary_labels, subset_size)
            balanced_datasets[label] = (balanced_features, balanced_binary_labels)

        return label_mapping, balanced_datasets    

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
        unique_labels = np.unique(labels)
        min_class_size = subset_size // 2

        balanced_features_list = []
        balanced_labels_list = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(label_indices, min_class_size, replace=len(label_indices) < min_class_size)
            balanced_features_list.append(features[sampled_indices])
            balanced_labels_list.append(labels[sampled_indices])

        balanced_features = np.concatenate(balanced_features_list)
        balanced_labels = np.concatenate(balanced_labels_list)

        return balanced_features, balanced_labels

    def objective(trial, balanced_features, balanced_labels):  # Removed label_mapping, updated parameters
        # Hyperparameters to optimize (no change)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        max_samples = trial.suggest_float('max_samples', 0.1, 0.5)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

        class_weights = calculate_class_weights(balanced_labels)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weights,
            random_state=seed,
            n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, balanced_features, balanced_labels, cv=skf, scoring='accuracy')
        mean_accuracy = scores.mean()

        # Log intermediate results
        logger.info(f"Trial {trial.number}: Mean accuracy = {mean_accuracy:.4f}")

        return mean_accuracy  # Return the mean accuracy directly


    def main():
        balanced_features, balanced_labels = load_and_balance_data(input_zarr_path, subset_size, seed)

        if balanced_features is None or balanced_labels is None:
            return  # Exit if data loading failed

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, balanced_features, balanced_labels), n_trials=num_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.params}")

        # Save the best parameters
        with open(best_params_path, 'w') as f:
            f.write(f"Best trial parameters:\n")
            f.write(f"Params: {study.best_trial.params}\n")
            f.write(f"Value: {study.best_trial.value}\n")
            f.write(f"Trial: {study.best_trial.number}\n")

        # Save the best model (without per-label separation)
        best_params = study.best_trial.params

        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_samples=best_params['max_samples'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            class_weight=calculate_class_weights(balanced_labels),
            random_state=seed,
            n_jobs=-1
        )
        
        model.fit(balanced_features, balanced_labels)
        joblib.dump(model, output_model_path)

        logger.info("Model training complete and saved.")

    main()


setup(
    group="cellcanvas",
    name="optimize-random-forest",
    version="0.0.5",
    title="Optimize Random Forest with Optuna on Zarr Data",
    description="A solution that optimizes a Random Forest model using Optuna, data from a Zarr zip store, and performs 10-fold cross-validation.",
    solution_creators=["Kyle Harrington"],
    tags=["random forest", "machine learning", "segmentation", "training", "cross-validation", "optuna"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "input_zarr_path", "type": "string", "required": True, "description": "Path to the input Zarr zip store containing the features and labels."},
        {"name": "output_model_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained Random Forest model."},
        {"name": "best_params_path", "type": "string", "required": True, "description": "Path for the output file containing the best parameters from Optuna."},
        {"name": "n_splits", "type": "string", "required": True, "description": "Number of splits for cross-validation."},
        {"name": "subset_size", "type": "string", "required": True, "description": "Total number of points for balanced subset."},
        {"name": "seed", "type": "string", "required": True, "description": "Random seed for reproducibility."},
        {"name": "num_trials", "type": "integer", "required": True, "description": "Number of Optuna trials to run."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
