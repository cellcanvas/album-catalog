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
  - pip:
    - album
    - copick
"""

def run():
    import os
    import joblib
    import numpy as np
    import zarr
    from zarr.storage import ZipStore
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import logging
    import copick
    import numcodecs

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_names = args.painting_segmentation_names.split(',')
    session_id = args.session_id
    user_id = args.user_id
    feature_types = args.feature_types.split(',')
    tomo_type = args.tomo_type
    voxel_spacing = float(args.voxel_spacing)
    eta = args.eta
    gamma = args.gamma
    max_depth = args.max_depth
    min_child_weight = args.min_child_weight
    max_delta_step = args.max_delta_step
    subsample = args.subsample
    colsample_bytree = args.colsample_bytree
    reg_lambda = args.reg_lambda
    reg_alpha = args.reg_alpha
    max_bin = args.max_bin
    class_weights_str = args.class_weights
    run_names = args.run_names.split(',') if args.run_names else None
    output_model_path = args.output_model_path

    # Load the Copick root from the configuration file
    logger.info(f"Loading Copick root configuration from: {copick_config_path}")
    root = copick.from_file(copick_config_path)
    logger.info("Copick root loaded successfully")

    def get_painting_segmentation(run, painting_name):
        try:
            segs = run.get_segmentations(
                user_id=user_id, is_multilabel=True, name=painting_name, voxel_size=voxel_spacing
            )
            if len(segs) == 0:
                logger.info(f"Segmentation does not exist seg name {painting_name}, user id {user_id}")
                return None
            else:
                seg = segs[0]
                group = zarr.open_group(seg.zarr(), mode="a")
                if '0' not in group:
                    shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
                    group.create_dataset('0', shape=shape, dtype=np.uint16, fill_value=0)
            return group['0']
        except (zarr.errors.PathNotFoundError, KeyError) as e:
            logger.error(f"Error opening painting segmentation zarr: {e}")
            return None

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def process_run(run, painting_segmentation_names, voxel_spacing, user_id, session_id, feature_types):
        all_painting_segs = []
        for painting_segmentation_name in painting_segmentation_names:
            painting_seg = get_painting_segmentation(run, painting_segmentation_name)
            if painting_seg:
                all_painting_segs.append(painting_seg[:])
        
        if not all_painting_segs:
            logger.info(f"No valid painting segmentations found for run {run}, skipping.")
            return None, None
        
        combined_labels = np.zeros_like(all_painting_segs[0])
        for painting_seg in all_painting_segs:
            combined_labels = np.where(painting_seg > 0, painting_seg, combined_labels)
        
        tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        if not tomo:
            logger.info(f"No tomogram found for run {run}, skipping.")
            return None, None

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
            return None, None

        concatenated_features = np.concatenate(all_features, axis=0)
        labels = combined_labels

        if labels.size == 0:
            logger.info(f"No labels found for run {run}, skipping.")
            return None, None

        flattened_labels = labels.reshape(-1)
        valid_indices = np.nonzero(flattened_labels > 0)[0]

        if valid_indices.size == 0:
            logger.info(f"No valid labels found for run {run}, skipping.")
            return None, None

        c, h, w, d = concatenated_features.shape
        reshaped_features = concatenated_features.reshape(c, h * w * d)

        filtered_features_list = [np.take(reshaped_features[i, :], valid_indices, axis=0) for i in range(c)]
        filtered_features = np.stack(filtered_features_list, axis=1)
        filtered_labels = flattened_labels[valid_indices] - 1

        logger.info(f"Processed run {run.name} with {filtered_labels.shape[0]} valid samples")
        return filtered_features, filtered_labels

    def load_features_and_labels_from_copick(root, run_names):
        features_list = []
        labels_list = []

        runs_to_process = [run for run in root.runs if run.name in run_names] if run_names else root.runs

        for run in runs_to_process:
            logger.info(f"Preparing run {run.name}")
            try:
                features, labels = process_run(run, painting_segmentation_names, voxel_spacing, user_id, session_id, feature_types)
                if features is not None and labels is not None:
                    features_list.append(features)
                    labels_list.append(labels)
                else:
                    logger.warning(f"No features/labels for {run.name}")
                    print(features)
                    print(labels)
            except Exception as e:
                logger.error(f"Error in processing run {run.name}: {e}")

        print(features_list)

        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        return all_features, all_labels

    # Extract training data from Copick runs
    logger.info("Extracting data from Copick runs...")
    features, labels = load_features_and_labels_from_copick(root, run_names)

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
    num_classes = len(unique_labels)  # Get the number of unique classes
    if class_weights_str:
        class_weights_list = list(map(float, class_weights_str.split(',')))
        class_weights = {label: (class_weights_list[i] if i < len(class_weights_list) else 0.0) for i, label in enumerate(unique_labels)}
    else:
        class_weights = calculate_class_weights(encoded_labels)
    logger.info(f"Class weights: {class_weights}")

    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[label] for label in encoded_labels])

    # Train XGBoost with 10-fold cross-validation
    logger.info(f"Training XGBoost with eta={eta}, gamma={gamma}, max_depth={max_depth}, min_child_weight={min_child_weight}, max_delta_step={max_delta_step}, subsample={subsample}, colsample_bytree={colsample_bytree}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha}, max_bin={max_bin}, and 10-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=10)
    scores = []
    
    params = {
        'eta': eta,
        'gamma': gamma,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'max_delta_step': max_delta_step,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'lambda': reg_lambda,
        'alpha': reg_alpha,
        'max_bin': max_bin,
        'objective': 'multi:softmax',
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'num_class': num_classes  # Specify the number of classes
    }
    
    for train_index, test_index in skf.split(features, encoded_labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights[train_index])
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        bst = xgb.train(params, dtrain, evals=[(dtest, 'eval')])
        preds = bst.predict(dtest)
        accuracy = np.mean(preds == y_test)
        scores.append(accuracy)
    
    scores = np.array(scores)
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean accuracy: {scores.mean()}, Std: {scores.std()}")

    # Train the final model on all data
    logger.info("Training final model on all data...")
    dtrain = xgb.DMatrix(features, label=encoded_labels, weight=sample_weights)
    final_model = xgb.train(params, dtrain)

    # Save the trained model and label encoder
    logger.info(f"Saving model and label encoder to {output_model_path}")
    joblib.dump((final_model, label_encoder), output_model_path)
    logger.info("Model and label encoder saved successfully")

setup(
    group="copick",
    name="train-model-xgboost-copick",
    version="0.0.2",
    title="Train XGBoost on Copick Data with Cross-Validation",
    description="A solution that processes Copick runs, filters runs with only one label, and trains an XGBoost model with 10-fold cross-validation.",
    solution_creators=["Kyle Harrington"],
    tags=["xgboost", "machine learning", "segmentation", "training", "cross-validation", "copick"],
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
        {"name": "run_names", "type": "string", "required": False, "description": "Comma-separated list of run names to process. If not provided, all runs will be processed."},
        {"name": "eta", "type": "float", "required": False, "description": "Step size shrinkage used in update to prevents overfitting.", "default": 0.3},
        {"name": "gamma", "type": "float", "required": False, "description": "Minimum loss reduction required to make a further partition on a leaf node of the tree.", "default": 0.0},
        {"name": "max_depth", "type": "integer", "required": False, "description": "The maximum depth of the trees.", "default": 6},
        {"name": "min_child_weight", "type": "float", "required": False, "description": "Minimum sum of instance weight needed in a child.", "default": 1.0},
        {"name": "max_delta_step", "type": "float", "required": False, "description": "Maximum delta step we allow each leaf output to be.", "default": 0.0},
        {"name": "subsample", "type": "float", "required": False, "description": "Subsample ratio of the training instances.", "default": 1.0},
        {"name": "colsample_bytree", "type": "float", "required": False, "description": "Subsample ratio of columns when constructing each tree.", "default": 1.0},
        {"name": "reg_lambda", "type": "float", "required": False, "description": "L2 regularization term on weights.", "default": 1.0},
        {"name": "reg_alpha", "type": "float", "required": False, "description": "L1 regularization term on weights.", "default": 0.0},
        {"name": "max_bin", "type": "integer", "required": False, "description": "Maximum number of discrete bins to bucket continuous features.", "default": 256},
        {"name": "class_weights", "type": "string", "required": False, "description": "Class weights for the XGBoost model as a comma-separated list.", "default": ""},
        {"name": "output_model_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained XGBoost model."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
