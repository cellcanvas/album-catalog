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
  - torch
  - torchvision
  - scikit-learn==1.3.2
  - joblib
  - h5py
  - imbalanced-learn
  - pip:
    - album
    - copick
"""

def run():
    import os
    import joblib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import zarr
    from zarr.storage import ZipStore
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from torch.utils.data import Dataset, DataLoader
    from copick.impl.filesystem import CopickRootFSSpec
    from imblearn.over_sampling import RandomOverSampler
    import random
    import logging

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
    torch.manual_seed(random_seed)

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

    # PyTorch Dataset and DataLoader
    class CustomDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x = torch.tensor(self.features[idx], dtype=torch.float32)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y

    # Simple PyTorch segmentation model
    class SegmentationModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SegmentationModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def train_torch_model(train_loader, class_weights, num_classes, device):
        model = SegmentationModel(train_loader.dataset.features.shape[1], num_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights.values())).to(device))

        model.train()
        for epoch in range(10):
            running_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader)}")

        # Save the trained model
        model_filename = os.path.join(output_dir, f"torch_model_step_{epoch}.pt")
        torch.save(model.state_dict(), model_filename)
        logger.info(f"Model saved as {model_filename}")

        return model

    def generate_prediction(model, features, shape, device):
        model.eval()
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(features_tensor)
        _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy().reshape(shape)

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for step, (train_index, _) in enumerate(sss.split(features, labels), 1):
        logger.info(f"Processing annotation step {step}/{num_annotation_steps}")

        selected_labels = labels[train_index]
        selected_features = features[train_index]

        logger.info("Oversampling minority classes")
        selected_features, selected_labels = oversample_minority_classes(selected_features, selected_labels)

        # Calculate class weights
        logger.info("Calculating class weights")
        class_weights = calculate_class_weights(selected_labels)

        # Create DataLoader
        train_dataset = CustomDataset(selected_features, selected_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Train PyTorch model
        logger.info("Training PyTorch model")
        model = train_torch_model(train_loader, class_weights, num_classes=len(np.unique(selected_labels)), device=device)

        # Print performance metrics
        logger.info(f"Performance metrics for step {step}:")
        predictions = generate_prediction(model, selected_features, segmentation.shape, device=device)
        print_performance_metrics(selected_labels, predictions)

    logger.info("Chunked annotation and PyTorch training completed successfully")


setup(
    group="cellcanvas",
    name="mock-annotation-torch",
    version="0.0.1",
    title="Mock Annotation and PyTorch Training on Copick Data",
    description="A solution that creates mock annotations based on multilabel segmentation, trains a PyTorch segmentation model in steps, and generates predictions.",
    solution_creators=["Kyle Harrington"],
    tags=["pytorch", "machine learning", "segmentation", "training", "copick", "mock annotation"],
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
