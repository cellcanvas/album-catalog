###album catalog: cellcanvas

from album.runner.api import setup, get_args

eval_env_file = """
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
    import json
    import joblib
    import numpy as np
    import zarr
    from sklearn.metrics import classification_report, jaccard_score, f1_score
    from copick.impl.filesystem import CopickRootFSSpec
    import xgboost as xgb
    import logging
    import copick.models

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = get_args()
    copick_config_path = args.copick_config_path
    voxel_spacing = float(args.voxel_spacing)
    tomo_type = args.tomo_type
    embedding_name = args.embedding_name
    input_user_id = args.input_user_id
    input_label_name = args.input_label_name
    run_name = args.run_name
    model_dir = args.model_dir
    output_dir = args.output_dir
    output_user_id = getattr(args, 'output_user_id', input_user_id)
    output_label_name = getattr(args, 'output_label_name', 'predicted_' + input_label_name)

    # Load Copick root
    root = CopickRootFSSpec.from_file(copick_config_path)

    # Get ground truth segmentation
    def get_segmentation(run, name):
        segs = run.get_segmentations(
            user_id=input_user_id, is_multilabel=True, name=name, voxel_size=voxel_spacing
        )
        if not segs:
            logger.info(f"No ground truth segmentation found: name {name}, user id {input_user_id}")
            return None
        seg = segs[0]
        return zarr.open(seg.zarr(), mode="r")['0'][:]

    # Get features
    def get_features(run):
        tomo = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        features = tomo.features
        if not features:
            logger.info(f"No features found for run {run}.")
            return None

        features_path = [f.path for f in features if f.feature_type == embedding_name]
        return zarr.open(features_path[0], mode="r")[:] if features_path else None

    # Get or create segmentation for saving predictions
    def get_or_create_segmentation(run, user_id, name, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, is_multilabel=True, name=name, voxel_size=voxel_spacing)
        if segs:
            seg = segs[0]
        else:
            seg = run.new_segmentation(voxel_spacing, name, session_id="0", is_multilabel=True, user_id=user_id)
        return seg

    # Load models, compute metrics, and save predictions
    def evaluate_models(features, ground_truth, segmentation_shape, run, output_user_id, output_label_name, voxel_spacing):
        results = {}
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl'):
                step = model_file.split('_')[-1].split('.')[0]
                model_path = os.path.join(model_dir, model_file)

                # Load model and encoder
                model, label_encoder = joblib.load(model_path)
                dtest = xgb.DMatrix(features)
                predictions = model.predict(dtest)
                predictions = label_encoder.inverse_transform(predictions.astype(int))

                # Reshape predictions to segmentation shape
                predicted_segmentation = predictions.reshape(segmentation_shape).astype(np.uint16)

                # Save predicted segmentation into Copick project
                output_segmentation_name = f"{output_label_name}{step}"
                seg = get_or_create_segmentation(run, output_user_id, output_segmentation_name, voxel_spacing)
                seg.from_numpy(predicted_segmentation, levels=1, dtype=np.uint16)
                logger.info(f"Predicted segmentation saved as {output_segmentation_name} for step {step}")

                # Calculate metrics
                iou = jaccard_score(ground_truth, predictions, average="weighted")
                f1 = f1_score(ground_truth, predictions, average="weighted")
                classification_stats = classification_report(ground_truth, predictions, output_dict=True)

                # Save metrics
                results[step] = {
                    "iou": iou,
                    "f1_score": f1,
                    "classification_report": classification_stats
                }
                logger.info(f"Metrics for step {step}: IoU={iou}, F1 Score={f1}")

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "evaluation_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {result_path}")

    # Run evaluation
    run = root.get_run(run_name)
    if not run:
        logger.error(f"Run {run_name} not found")
        return

    segmentation = get_segmentation(run, input_label_name)
    features = get_features(run)

    if segmentation is None or features is None:
        logger.error("Failed to load segmentation or features")
        return

    segmentation_shape = segmentation.shape
    features_reshaped = features.reshape(features.shape[0], -1).T

    evaluate_models(features_reshaped, segmentation.reshape(-1), segmentation_shape, run, output_user_id, output_label_name, voxel_spacing)

setup(
    group="cellcanvas",
    name="mock-annotation-evaluation",
    version="0.0.3",
    title="Model Evaluation on Copick Data",
    description="Evaluates segmentation models from the mock-annotation solution on Copick data, generating metrics like IoU and F1, and saves the predicted segmentation into the Copick project.",
    solution_creators=["Kyle Harrington"],
    tags=["xgboost", "machine learning", "segmentation", "evaluation", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "voxel_spacing", "type": "float", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "tomo_type", "type": "string", "required": True, "description": "Tomogram type to use for each tomogram."},
        {"name": "embedding_name", "type": "string", "required": True, "description": "Name of the embedding features to use."},
        {"name": "input_user_id", "type": "string", "required": True, "description": "User ID for the input segmentation."},
        {"name": "input_label_name", "type": "string", "required": True, "description": "Name of the input label segmentation."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the run to process."},
        {"name": "model_dir", "type": "string", "required": True, "description": "Directory of trained model files."},
        {"name": "output_dir", "type": "string", "required": True, "description": "Directory to save evaluation results."},
        {"name": "output_user_id", "type": "string", "required": False, "description": "User ID for the output predicted segmentation."},
        {"name": "output_label_name", "type": "string", "required": False, "description": "Name of the output predicted segmentation."}
    ],
    run=run,
    dependencies={
        "environment_file": eval_env_file
    },
)
