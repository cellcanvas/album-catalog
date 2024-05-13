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
  - scikit-learn==1.3.2
  - joblib
  - pip:
    - album
    - copick
    - cellcanvas
"""

def run():
    import os
    import joblib
    import numpy as np
    import zarr
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from copick.impl.filesystem import CopickRootFSSpec
    from cellcanvas.data.data_set import DataSet
    from cellcanvas.data.data_manager import DataManager
    from cellcanvas.semantic.segmentation_manager import (
        SemanticSegmentationManager,
    )
    import time

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = int(args.voxel_spacing)
    model_output_path = args.model_output_path
    n_estimators = int(args.n_estimators)

    # Load the Copick root from the configuration file
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "painting_segmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run):
        segs = run.get_segmentations(
            user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing
        )
        if len(segs) == 0:
            print(f"No existing segmentation found for '{painting_segmentation_name}', creating new segmentation.")
            if not run.get_voxel_spacing(voxel_spacing):
                print(f"Voxel spacing: {voxel_spacing} does not exist.")
                return None
            seg = run.new_segmentation(voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id)
            tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised")
            if not tomogram:
                print("No tomogram 'denoised' available.")
                return None
            shape = zarr.open(tomogram.zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
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

    def train_random_forest(features, labels, class_weights):
        """Train a Random Forest Classifier on the features and labels."""
        print(f"Training Random Forest with {len(class_weights)} classes...")
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            max_depth=15,
            max_samples=0.05,
            class_weight=class_weights
        )
        model.fit(features, labels)
        elapsed_time = time.time() - start_time
        print(f"Random Forest trained in {elapsed_time:.2f} seconds")
        return model

    def save_model(model, model_output_path):
        """Save the trained Random Forest model using joblib."""
        print(f"Saving model to {model_output_path}")
        joblib.dump(model, model_output_path)
        print("Model saved successfully")

    def get_embedding_zarr(run):
        """Retrieve the denoised tomogram embeddings."""
        return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"]

    # Function to load datasets from Copick runs
    def load_datasets_from_copick(root, max_runs: int = 50) -> DataManager:
        datasets = []
        for idx, run in enumerate(root.runs[:max_runs]):
            print(f"Processing run {idx + 1}/{len(root.runs)}: {run}")
            painting_seg = get_painting_segmentation(run)
            if not painting_seg:
                print("Painting segmentation failed, skipping.")
                continue

            embedding_zarr = get_embedding_zarr(run)

            # Mock paths for features, labels, and segmentation (adjust paths as per your specific requirements)
            image_path = run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").static_path
            if len(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").features) == 0:
                print("Missing features.")
                continue
            features_path = run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").features[0].path        
            labels_path = os.path.join(run.static_path, "Segmentations/10.000_kish_17006_batchpainttest01-multilabel.zarr")
            segmentation_path = os.path.join(run.static_path, "Segmentations/10.000_kish_17006_cellcanvasrcr01-multilabel.zarr")

            # Create and append the DataSet object
            dataset = DataSet.from_paths(image_path, features_path, labels_path, segmentation_path)
            datasets.append(dataset)

        return DataManager(datasets)

    # Extract training data using DataManager
    print("Extracting data from Copick runs...")
    data_manager = load_datasets_from_copick(root)
    features_all, labels_all = data_manager.get_training_data()

    print(f"Total samples: {features_all.shape[0]}, Total features per sample: {features_all.shape[1]}")

    if labels_all.size == 0:
        print("No labels present. Skipping model update.")
    else:
        # Calculate class weights
        class_weights = calculate_class_weights(labels_all)
        print(f"Class balance calculated: {class_weights}")

        # Update model using SegmentationManager
        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            max_depth=12,
            max_samples=0.05,
            max_features='sqrt',
            class_weight='balanced'
        )

        segmentation_manager = SemanticSegmentationManager(data=data_manager, model=clf)
        print("Starting to fit model")
        segmentation_manager.fit()

        # Save the trained model
        save_model(segmentation_manager.model, model_output_path)

        print(f"Random Forest model trained and saved to {model_output_path}")

setup(
    group="cellcanvas",
    name="train-model",
    version="0.0.5",
    title="Train Random Forest on Copick Painted Segmentation Data",
    description="A solution that trains a Random Forest model using Copick painted segmentation data and exports the trained model.",
    solution_creators=["Kyle Harrington"],
    tags=["random forest", "machine learning", "segmentation", "training", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_name", "type": "string", "required": False, "description": "Name for the painting segmentation."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "n_estimators", "type": "string", "required": True, "description": "Number of trees in the Random Forest."},
        {"name": "model_output_path", "type": "string", "required": True, "description": "Path for the output joblib file containing the trained Random Forest model"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
