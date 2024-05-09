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
"""

def run():
    import joblib
    import numpy as np
    import zarr
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = args.voxel_spacing
    model_output_path = args.model_output_path

    # Load the Copick root from the configuration file
    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_painting_segmentation_name(painting_name):
        return painting_name if painting_name else "painting_segmentation"

    painting_segmentation_name = get_painting_segmentation_name(painting_segmentation_name)

    def get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
        if len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    def extract_features_labels(painting_seg_array, embedding_zarr):
        """Extract features and labels from the segmentation and embeddings arrays."""
        mask = painting_seg_array != 0
        labels = painting_seg_array[mask].ravel() - 1  # Adjust labels to start from 0
        features = embedding_zarr[mask].reshape(-1, embedding_zarr.shape[-1])

        return features, labels

    def calculate_class_weights(labels):
        """Calculate class weights for balancing the Random Forest model."""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        return dict(zip(unique_labels, class_weights))

    def train_random_forest(features, labels, class_weights):
        """Train a Random Forest Classifier on the features and labels."""
        model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            max_depth=15,
            max_samples=0.05,
            class_weight=class_weights
        )
        model.fit(features, labels)
        return model

    def save_model(model, model_output_path):
        """Save the trained Random Forest model using joblib."""
        joblib.dump(model, model_output_path)

    def get_embedding_zarr(run, voxel_spacing):
        """Retrieve the denoised tomogram embeddings."""
        return zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"]

    # Loop through each run in the copick project
    features_list, labels_list = [], []
    for run in root.runs:
        painting_seg = get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing)
        embedding_zarr = get_embedding_zarr(run, voxel_spacing)
        features, labels = extract_features_labels(painting_seg, embedding_zarr)
        features_list.append(features)
        labels_list.append(labels)

    # Concatenate all features and labels
    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    if labels_all.size == 0:
        print("No labels present. Skipping model update.")
        return

    # Calculate class weights
    class_weights = calculate_class_weights(labels_all)
    print(f"Class balance calculated {class_weights}")

    # Train the Random Forest model
    model = train_random_forest(features_all, labels_all, class_weights)
    save_model(model, model_output_path)

    print(f"Random Forest model trained and saved to {model_output_path}")
 
setup(
    group="cellcanvas",
    name="train-model",
    version="0.0.1",
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
