###album catalog: cellcanvas

from io import StringIO

from album.runner.api import setup

env_file = StringIO(
    """channels:
  - pytorch-nightly
  - fastai
  - conda-forge
  - defaults    
dependencies:
  - python>=3.8
  - pybind11
  - pip
  - boost-cpp
  - numpy >= 1.26
  - scipy
  - numba
  - scikit-image
  - matplotlib
  - pandas
  - pytables
  - jupyter
  - notebook
  - jupytext
  - quantities
  - ipywidgets
  - meshio
  - zarr
  - xarray
  - hdf5
  - mpfr
  - gmp
  - reikna
  - jupyterlab
  - dill    
  - einops
  - fire
  - maven
  - pillow
  - openjpeg
  - imagecodecs
  - "bokeh>=2.4.2,<3"
  - python-graphviz
  - ipycytoscape
  - fftw
  - s3fs
  - pooch
  - yappi
  - ftfy
  - tqdm >= 4.38
  - imageio
  - pyarrow >= 13
  - squidpy
  - h5py
  - tifffile
  - nilearn
  - flake8
  - pytest
  - asv
  - pint
  - pytest-cov
  - mypy
  - flask
  - libnetcdf
  - ruff
  - confuse
  - appdirs
  - labeling >= 0.1.12
  - lazy_loader
  - lxml
  - ninja
  - gql
  - boto3
  - album
  - "opencv-python-headless>=0.4.8"
  - "transformers>=4.36.1"
  - "imageio-ffmpeg>=0.4.8"
  - networkx
  - ipython      
  - pip:
    - imaris-ims-file-reader
    - scanpy
    - "tensorstore>=0.1.51"
    - compressed-segmentation
    - pyspng-seunglab
    - "pyheif>=0.7"
    - "ome-zarr>=0.8.0"
    - epc
    - pygeodesic
    - skan
    - "pydantic-ome-ngff>=0.2.3"
    - "python-dotenv>=0.21"
    - validate-pyproject[all]
    - ndjson
    - requests_toolbelt
    - networkx
    - "xgboost>=2"
    - cryoet-data-portal>=2
    - mrcfile
    - "starfile>=0.5.0"
    - "imodmodel>=0.0.7"
    - cryotypes
"""
)


def run():
    import IPython

    IPython.start_ipython()


setup(
    group="ux_evaluation_winter2024",
    name="python-nographics",
    version="0.0.2",
    title="Parent environment for supporting the Winter 2024 UX Evaluation.",
    description="Parent environment for supporting the Winter 2024 UX Evaluation",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle. Also check out TeamTomo", "url": ""}],
    tags=["imaging", "cryoet", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={"environment_file": env_file},
)
