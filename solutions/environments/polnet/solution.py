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
  - python=3.10
  - mrcfile
  - "numpy<2"
  - pip
  - zarr
  - scipy
  - jax
  - vtk
  - pandas
  - pynrrd
  - scikit-image
  - jupyter
  - wget
  - ipyfilechooser
  - cudatoolkit=11.2
  - cudnn>=8.1.0
  - pip:
    - git+https://github.com/kephale/mrc2omezarr
    - album
    - "git+https://github.com/uermel/copick.git"
"""


def download_file(url, destination):    
    """Download a file from a URL to a destination path."""
    import requests
    
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def install():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/anmartinezs/polnet"])


setup(
    group="environments",
    name="polnet",
    version="0.0.3",
    title="An environment to support polnet and copick",
    description="An album solution for copick polnet projects .",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["Python", "environment"],
    license="MIT",
    covers=[],
    install=install,
    album_api_version="0.5.1",
    dependencies={"environment_file": env_file},
)
