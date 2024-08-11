###album catalog: cellcanvas

from io import StringIO
from album.runner.api import setup
import tempfile
import os

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python==3.9
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit
  - pytorch-cuda
  - dask
  - einops
  - h5py
  - magicgui
  - monai
  - numpy<2
  - pytorch-lightning
  - qtpy
  - rich
  - scikit-image
  - scipy
  - tensorboard
  - mlflow
  - mrcfile
  - pip:
    - git+https://github.com/kephale/morphospaces.git@copick
    - git+https://github.com/copick/copick.git
    - git+https://github.com/kephale/copick-torch.git
"""

setup(
    group="environments",
    name="copick-monai",
    version="0.0.3",
    title="An environment to support copick monai projects",
    description="An album solution for copick monai morphospaces projects .",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["Python", "environment"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    dependencies={"environment_file": env_file},
)
