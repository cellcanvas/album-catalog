###album catalog: cellcanvas

from album.runner.api import get_args, setup

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python==3.9
  - numpy==1.26.3
  - pip
  - pip:
    - tqdm
    - torch
    - tensorboard==2.17.0
    - zarr
"""

def run():
    import zarr
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import subprocess

    # CLI arguments
    args = get_args()
    channel_first = args.channel_first
    embedding_path = args.embeddings
    label_path = args.labels
    embed_zarr_key = args.embed_zarr_key
    label_zarr_key = args.label_zarr_key
    k = args.k
    logdir = args.logdir
    label_img = args.label_img
    port = args.port

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(logdir)
    print(f'embed_zarr_key {embed_zarr_key}')
    if embed_zarr_key:
        embeddings = np.load(embedding_path) if embedding_path.endswith(".npy") else zarr.open(embedding_path, mode='r')[embed_zarr_key][:]
    else:
        embeddings = np.load(embedding_path) if embedding_path.endswith(".npy") else zarr.open(embedding_path, mode='r')[:]
    
    print(f'label_zarr_key {label_zarr_key}')
    if label_zarr_key:
        labels = np.load(label_path) if label_path.endswith(".npy") else zarr.open(label_path, mode='r')[label_zarr_key][:]
    else:
        labels = np.load(label_path) if label_path.endswith(".npy") else zarr.open(label_path, mode='r')[:]

    print(f'label_img {type(label_img)}')
    if label_img:
        label_img = torch.Tensor(np.load(label_img))
    else:
        label_img = None 
    
    if not channel_first:
        embeddings = embeddings.T
    
    embeddings = np.moveaxis(embeddings, 0, -1)
    unique_labels = [] if labels is None else np.unique(labels)
    embd_list = []
    label_list = []
    for label in tqdm(unique_labels):
        arr = embeddings[labels==label]
        # Calculate the number of indices to select (10% of the array length)
        num_indices = min(1000, round(len(arr) * k))
        # Randomly choose the indices
        selected_indices = np.random.choice(len(arr), num_indices, replace=False)
        for i in selected_indices:
            embd_list.append(arr[i])
        label_list = label_list + [label]*num_indices

    # log embeddings
    writer.add_embedding(np.array(embd_list),
                         metadata=np.array(label_list),
                         label_img=label_img)
    writer.close()


    # Launch Tesnsorboard
    bash_command = f'tensorboard --logdir="{logdir}" --port {port}'
    print(bash_command)
    subprocess.run(bash_command, shell=True, stdout = subprocess.DEVNULL, executable="/bin/bash")

setup(
    group="cellcanvas",
    name="embedding-projector",
    version="0.0.2",
    title="Generate a Tensorboard projector for visualzing the embeddings",
    description="Automatically generate Tensorboard event files and launch a visualizer for it. Currently suppot zarr and npy files.",
    solution_creators=["Zhuowen Zhao"],
    cite=[{"text": "Tensorboard Team.", "url": "https://www.tensorflow.org/tensorboard"}],
    tags=["tensorboard", "embedding", "visualization", "Python"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "channel_first",
            "description": "Channel first for the embedding vectors. Default is True.",
            "type": "boolean",
            "required": False,
            "default": True,
        },
        {
            "name": "embeddings",
            "description": "Path to the feature vector embeddings",
            "type": "string",
            "required": True,
        },
        {
            "name": "labels",
            "description": "Path to the labels correspond to the embeddings. Labels should be (N,) shape or a flattened array.",
            "type": "string",
            "required": False,
        },
        {
            "name": "embed_zarr_key",
            "description": "Key to get the embeding zarr file.",
            "type": "string",
            "required": False,
        },
        {
            "name": "label_zarr_key",
            "description": "Key to get the label zarr file.",
            "type": "string",
            "required": False,
        },
        {
            "name": "label_img",
            "description": "Path to the npy image files corresponds to the embeddings.",
            "type": "string",
            "required": False,
        },
        {
            "name": "logdir",
            "description": "Output directory name in the current working directory/runs. Default is runs/tensorboard",
            "type": "string",
            "required": False,
            "default": "runs/tensorboard",
        },
        {
            "name": "k",
            "description": "Percentage of the features to visualize (<= 1000). Default is 0.01",
            "type": "float",
            "required": False,
            "default": 0.01,
        },
        {
            "name": "port",
            "description": "Port number to launch Tensorboard server. Default is 6006",
            "type": "string",
            "required": False,
            "default": "6006",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)