###album catalog: cellcanvas

from io import StringIO

from album.runner.api import setup

env_file = StringIO(
    """name: napryo
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
    - git+https://github.com/cellcanvas/cellcanvas
    - git+https://github.com/napari/napari.git@4f4c063ae5dd79d6d188e201d44b8d57eba71909
"""
)


def run():
    from album.runner.api import get_args
    from cellcanvas import CellCanvasApp
    import napari
    import appdirs
    from pathlib import Path
    import zipfile

    dataset_name = get_args().dataset_name
    
    all_datasets = ["cellcanvas_crop_007", "cellcanvas_crop_009", "cellcanvas_crop_010",]

    if dataset_name not in all_datasets:
        print(f"Dataset {dataset_name} invalid. Select from {all_datasets}")
        return
        
    data_directory = Path(appdirs.user_data_dir("cellcanvas"))

    zip_path = data_directory / f"{dataset_name}.zarr.zip"

    zarr_directory = data_directory / f"{dataset_name}.zarr"

    # Unzip the Zarr archive if the directory doesn't exist
    if not zarr_directory.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Unzipping {zip_path} to {zarr_directory}")
            zip_ref.extractall(data_directory)
    
    CellCanvasApp(zarr_directory)
    
    napari.run()


setup(
    group="ux_evaluation_winter2024",
    name="run_evaluation",
    version="0.0.1",
    title="First portable CellCanvas demo.",
    description="First portable CellCanvas demo",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://cellcanvas.org"}],
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
            {"name": "dataset_name", "type": "string", "required": True, "description": "Name of dataset for evaluation: cellcanvas_crop_007, cellcanvas_crop_009, or cellcanvas_crop_010", "default": "cellcanvas_crop_007"},
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
