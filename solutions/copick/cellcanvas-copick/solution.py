###album catalog: cellcanvas

from io import StringIO

from album.runner.api import setup, get_data_path

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
    - lxml_html_clean    
    - omero-rois
    - "tensorstore>=0.1.51"
    - "opencv-python-headless>=0.4.8"
    - "ome-zarr>=0.8.0"
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
    - git+https://github.com/cellcanvas/surforama    
    - "git+https://github.com/cellcanvas/cellcanvas@lazy-large"
    - git+https://github.com/napari/napari.git
    - git+https://github.com/uermel/copick.git
"""
)

def install():
    import os
    import platform
    import subprocess

    # Determine the operating system
    os_name = platform.system()
    
    def find_available_package_manager():
        # Define the order of preference for package managers
        managers = ["micromamba", "mamba", "conda"]

        for manager in managers:
            try:
                # Check if the package manager is available by running its version command
                subprocess.run([manager, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return manager  # Return the first available package manager
            except subprocess.CalledProcessError:
                continue
            except FileNotFoundError:
                continue

        raise RuntimeError("No suitable package manager found (micromamba, mamba, or conda).")    

    try:
        package_manager = find_available_package_manager()
        os_name = platform.system()

        if os_name == "Darwin":  # macOS
            subprocess.run([package_manager, "install", "-c", "conda-forge", "ocl_icd_wrapper_apple"], check=True)
        elif os_name == "Linux":
            subprocess.run([package_manager, "install", "-c", "conda-forge", "ocl-icd-system"], check=True)
        else:
            # Raise an exception for unsupported operating systems
            raise EnvironmentError(f"Unsupported operating system: {os_name}")
    except Exception as e:
        # Raise the exception to be caught by higher-level error handling logic if necessary
        raise RuntimeError(f"Installation failed: {e}")

def run():
    from album.runner.api import get_args
    from cellcanvas import CellCanvasApp
    import napari
    import sys
    from copick.impl.filesystem import CopickRootFSSpec
    from cellcanvas._copick.widget import NapariCopickExplorer

    # "/Volumes/kish@CZI.T7/demo_project/copick_config_kyle.json"
    copick_config = get_args().copick_config
    
    root = CopickRootFSSpec.from_file(copick_config)
    # root = CopickRootFSSpec.from_file("/Volumes/kish@CZI.T7/chlamy_copick/copick_config_kyle.json")
        
    viewer = napari.Viewer()

    # Hide layer list and controls
    # viewer.window.qt_viewer.dockLayerList.setVisible(False)
    # viewer.window.qt_viewer.dockLayerControls.setVisible(False)

    copick_explorer_widget = NapariCopickExplorer(viewer, root)
    viewer.window.add_dock_widget(copick_explorer_widget, name="Copick Explorer", area="left")
    
    napari.run()


setup(
    group="copick",
    name="cellcanvas-copick",
    version="0.0.5",
    title="Run CellCanvas with a copick project.",
    description="Run CellCanvas with a copick project",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas team.", "url": "https://cellcanvas.org"}],
    tags=["imaging", "cryoet", "Python", "napari", "cellcanvas", "copick"],
    license="MIT",
    covers=[
        {
            "description": "CelCanvas screenshot.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config", "type": "string", "required": True, "description": "Path to the copick project config file."},
    ],
    run=run,
    install=install,
    dependencies={"environment_file": env_file},
)
