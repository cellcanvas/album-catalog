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
    from typing import Protocol
    import time

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
    model_output_path = args.model_output_path
    n_estimators = int(args.n_estimators)

    # Load the Copick root from the configuration file
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "paintingsegmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run):
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

    def save_features_and_labels(features, labels, model_output_path):
        """Save features and labels to files."""
        features_path = model_output_path.replace('.joblib', '_features.npy')
        labels_path = model_output_path.replace('.joblib', '_labels.npy')

        print(f"Saving features to {features_path}")
        np.save(features_path, features)
        print(f"Saving labels to {labels_path}")
        np.save(labels_path, labels)
        
    def get_embedding_zarr(run):
        """Retrieve the denoised tomogram embeddings."""
        return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"]

    # Function to load features and labels from Copick runs
    def load_features_and_labels_from_copick(root, max_runs: int = 50):
        all_features = []
        all_labels = []

        for idx, run in enumerate(root.runs[:max_runs]):
            print(f"Processing run {idx + 1}/{len(root.runs)}: {run}")
            painting_seg = get_painting_segmentation(run)
            if not painting_seg:
                print("Painting segmentation failed, skipping.")
                continue

            embedding_zarr = get_embedding_zarr(run)

            if len(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").features) == 0:
                print("Missing features.")
                continue

            features = np.array(zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").features[0].path, "r"))
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

            all_features.append(filtered_features)
            all_labels.append(filtered_labels)

            print(f"Found new labels {filtered_labels.shape}")

        if len(all_features) > 0 and len(all_labels) > 0:
            all_features = np.concatenate(all_features)
            all_labels = np.concatenate(all_labels)
        return all_features, all_labels

    # Extract training data from Copick runs
    print("Extracting data from Copick runs...")
    features_all, labels_all = load_features_and_labels_from_copick(root)

    if features_all and labels_all:
        print(f"Total samples: {features_all.shape[0]}, Total features per sample: {features_all.shape[1]}")
    else:
        print("No features.")
        return
    
    if labels_all.size == 0:
        print("No labels present. Skipping model update.")
    else:
        print("Saving features")
        save_features_and_labels(features_all, labels_all, model_output_path)
        
        # Calculate class weights
        class_weights = calculate_class_weights(labels_all)
        print(f"Class balance calculated: {class_weights}")

        # Train the Random Forest model
        model = train_random_forest(features_all, labels_all, class_weights)

        # Save the trained model
        save_model(model, model_output_path)        

        print(f"Random Forest model trained and saved to {model_output_path}")

setup(
    group="cellcanvas",
    name="train-model",
    version="0.0.11",
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
