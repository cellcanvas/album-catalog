###album catalog: cellcanvas

from io import StringIO

from album.runner.api import setup

env_file = StringIO(
    """name: cellcanvas
channels:
  - pytorch-nightly
  - fastai
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pybind11
  - pip
  - boost-cpp
  - mpfr
  - gmp
  - cgal
  - numpy
  - scipy
  - scikit-image
  - scikit-learn
  - matplotlib
  - pandas
  - pytables
  - jupyter
  - notebook
  - jupytext
  - quantities
  - ipywidgets
  - vispy
  - meshio
  - zarr
  - xarray
  - hdf5
  - mpfr
  - gmp
  - pyqt
  - omero-py
  - pyopencl
  - reikna
  - jupyterlab
  - pytorch
  - einops
  - fire
  - pillow
  - openjpeg
  - imagecodecs
  - "bokeh>=2.4.2,<3"
  - python-graphviz
  - fftw
  - napari-segment-blobs-and-things-with-membranes
  - s3fs
  - fsspec
  - pooch
  - qtpy
  - superqt
  - yappi
  - ftfy
  - tqdm
  - imageio
  - pyarrow
  - squidpy
  - h5py
  - tifffile
  - nilearn
  - flake8
  - pytest
  - asv
  - pint
  - pytest-qt
  - pytest-cov
  - mypy
  - opencv
  - flask
  - libnetcdf
  - ruff
  - confuse
  - labeling >= 0.1.12
  - lazy_loader
  - lxml
  - ninja
  - pythran
  - gql
  - boto3
  - appdirs
  - album
  - "imageio-ffmpeg>=0.4.8"
  - networkx
  - ipython  
  - pip:
    - idr-py
    - omero-rois
    - "tensorstore>=0.1.51"
    - "opencv-python-headless>=0.4.8"    
    - "ome-zarr>=0.8.0"
    - tootapari
    - Mastodon.py
    - pygeodesic
    - "pydantic-ome-ngff>=0.2.3"
    - "python-dotenv>=0.21"
    - validate-pyproject[all]
    - ndjson
    - requests_toolbelt
    - "xgboost>=2"
    - "cryoet-data-portal>=2"
    - "napari-cryoet-data-portal>=0.2.1"
    - mrcfile
    - "starfile>=0.5.0"
    - "imodmodel>=0.0.7"
    - cryotypes
    - blik
    - napari-properties-plotter
    - napari-properties-viewer
    - napari-label-interpolator
    - git+https://github.com/cellcanvas/surforama@b764471bd489a1805fe8b24567176922cb0609dd    
    - git+https://github.com/cellcanvas/cellcanvas@ec6e6768e7efa216a55e477051deff6e064264bf    
    - git+https://github.com/napari/napari.git@bd82daf209e17e6192045bb290c125e7edde40cb
"""
)


def run():
    from album.runner.api import get_args
    from cellcanvas import CellCanvasApp
    import napari
    import appdirs
    from pathlib import Path
    import zipfile
    import logging
    import sys

    viewer = napari.Viewer()
    
    napari.run()


setup(
    group="cellcanvas",
    name="cellcanvas",
    version="0.0.1",
    title="Run CellCanvas.",
    description="Run CellCanvas",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas team.", "url": "https://cellcanvas.org"}],
    tags=["imaging", "cryoet", "Python", "napari", "cellcanvas"],
    license="MIT",
    covers=[
        {
            "description": "CelCanvas screenshot.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    args=[
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
