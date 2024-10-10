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
  - imbalanced-learn  
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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    import xgboost as xgb
    import logging
    from copick.impl.filesystem import CopickRootFSSpec
    from imblearn.over_sampling import RandomOverSampler
    import random

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    tomo_type = args.tomo_type
    embedding_name = args.embedding_name
    input_session_id = args.input_session_id
    input_user_id = args.input_user_id
    input_label_name = args.input_label_name
    num_annotation_steps = int(args.num_annotation_steps)
    run_name = args.run_name
    random_seed = args.random_seed
    output_dir = args.output_dir

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the Copick root from the configuration file
    logger.info(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    logger.info("Copick root loaded successfully")

    def get_segmentation(run, name):
        try:
            segs = run.get_segmentations(
                user_id=input_user_id, is_multilabel=True, name=name, voxel_size=voxel_spacing
            )
            if len(segs) == 0:
                logger.info(f"Segmentation does not exist: name {name}, user id {input_user_id}")
                return None
            seg = segs[0]
            return zarr.open(seg.zarr(), mode="r")['0'][:]
        except Exception as e:
            logger.error(f"Error opening segmentation zarr: {e}")
            return None

    def get_features(run):
        try:
            tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
            if not tomo:
                logger.info(f"No tomogram found for run {run}, skipping.")
                return None

            features = tomo.features
            if len(features) == 0:
                logger.info(f"No features found for run {run}, skipping.")
                return None

            features_path = [f.path for f in features if f.feature_type == embedding_name]
            if len(features_path) == 0:
                logger.info(f"No {embedding_name} features found for run {run}, skipping.")
                return None
            
            return zarr.open(features_path[0], mode="r")[:]
        except Exception as e:
            logger.error(f"Error opening features zarr: {e}")
            return None

    def calculate_class_weights(labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = {label: total_samples / (len(unique_labels) * count) for label, count in zip(unique_labels, counts)}
        return class_weights

    def train_xgboost_model(X_train, y_train, class_weights):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        
        sample_weights = np.array([class_weights[label] for label in y_train])
        
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded, weight=sample_weights)
        
        params = {
            'objective': 'multi:softmax',
            'tree_method': 'gpu_hist',
            'eval_metric': 'mlogloss',
            'num_class': len(np.unique(y_train_encoded)),
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1  # This helps with imbalanced classes
        }
        
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Save the trained model and label encoder together
        model_filename = os.path.join(output_dir, f"xgboost_model_with_encoder_step_{step}.pkl")
        joblib.dump((model, label_encoder), model_filename)
        logger.info(f"Model and LabelEncoder saved as {model_filename}")

        return model, label_encoder, y_pred

    def generate_prediction(model, features, shape, label_encoder):
        dtest = xgb.DMatrix(features.reshape(-1, features.shape[0]))
        predictions = model.predict(dtest)
        
        # Convert predicted labels back to the original label set
        predictions_original_labels = label_encoder.inverse_transform(predictions.astype(int))
        
        # Reshape the predictions to match the segmentation shape
        return predictions_original_labels.reshape(shape)

    def oversample_minority_classes(X, y):
        oversample = RandomOverSampler(sampling_strategy='not majority', random_state=random_seed)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
        return X_resampled, y_resampled

    def print_performance_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_true, y_pred))        

    # Get the run
    run = root.get_run(run_name)
    if not run:
        logger.error(f"Run {run_name} not found")
        return

    # Get segmentation and features
    segmentation = get_segmentation(run, input_label_name)
    features = get_features(run)

    if segmentation is None or features is None:
        logger.error("Failed to load segmentation or features")
        return

    # Prepare data
    labels = segmentation.reshape(-1)
    features = features.reshape(features.shape[0], -1).T

    # Split the labels into chunks
    chunk_size = 10000  # Set your desired chunk size
    label_chunks = np.array_split(labels, len(labels) // chunk_size)
    feature_chunks = np.array_split(features, len(features) // chunk_size)

    # Shuffle the chunks
    indices = list(range(len(label_chunks)))
    random.shuffle(indices)

    # Calculate the number of chunks per step
    chunks_per_step = len(label_chunks) // num_annotation_steps

    # Use StratifiedShuffleSplit for balanced sampling
    sss = StratifiedShuffleSplit(n_splits=num_annotation_steps, test_size=0.8, random_state=random_seed)
    
    for step, (train_index, _) in enumerate(sss.split(features, labels), 1):
        logger.info(f"Processing annotation step {step}/{num_annotation_steps}")

        selected_labels = labels[train_index]
        selected_features = features[train_index]

        logger.info("Oversampling minority classes")
        selected_features, selected_labels = oversample_minority_classes(selected_features, selected_labels)

        # Calculate class weights
        logger.info("Calculating class weights")
        class_weights = calculate_class_weights(selected_labels)

        # Train XGBoost model
        logger.info("Training model")
        model, label_encoder, y_pred = train_xgboost_model(selected_features, selected_labels, class_weights)

        # Print performance metrics
        logger.info(f"Performance metrics for step {step}:")
        print_performance_metrics(selected_labels, y_pred)

    logger.info("Chunked annotation and XGBoost training completed successfully")


setup(
    group="cellcanvas",
    name="mock-annotation",
    version="0.0.9",
    title="Mock Annotation and XGBoost Training on Copick Data",
    description="A solution that creates mock annotations based on multilabel segmentation, trains XGBoost models in steps, and generates predictions.",
    solution_creators=["Kyle Harrington"],
    tags=["xgboost", "machine learning", "segmentation", "training", "copick", "mock annotation"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram."},
        {"name": "embedding_name", "type": "string", "required": True, "description": "Name of the embedding features to use."},
        {"name": "input_session_id", "type": "string", "required": True, "description": "Session ID for the input segmentation."},
        {"name": "input_user_id", "type": "string", "required": True, "description": "User ID for the input segmentation."},
        {"name": "input_label_name", "type": "string", "required": True, "description": "Name of the input label segmentation."},
        {"name": "num_annotation_steps", "type": "integer", "required": True, "description": "Number of annotation steps to perform."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the run to process."},
        {"name": "output_dir", "type": "string", "required": True, "description": "Directory to save trained models."},
        {"name": "random_seed", "type": "integer", "required": False, "default": 17171, "description": "Random seed for reproducibility."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
