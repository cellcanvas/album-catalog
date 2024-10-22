###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - napari
  - pyqt
  - numpy
  - paramiko
  - requests
  - smbprotocol
  - pip:
    - album
    - "copick[all]"
    - "git+https://github.com/cellcanvas/napari-cellcanvas.git@experimental-server"
    - "sshfs>=2024.6.0"
"""

def run():
    from napari import Viewer
    # from napari_cellcanvas import CellCanvasWidget
    from napari_cellcanvas import ExperimentalCellCanvasWidget
    import napari
    import requests
    import json
    import os

    # Retrieve the arguments passed to the solution
    args = get_args()

    copick_config_path = args.copick_config_path

    # Initialize the Napari viewer
    viewer = Viewer()

    # Initialize the CellCanvasWidget with the provided arguments
    widget = ExperimentalCellCanvasWidget(viewer=viewer, copick_config_path=copick_config_path)

    # Add the widget to the viewer's dock
    viewer.window.add_dock_widget(widget, area='right')

    # Start Napari
    napari.run()

setup(
    group="cellcanvas",
    name="napari-cellcanvas",
    version="0.0.7",
    title="napari-CellCanvas",
    description="A solution that launches napari-cellcanvas with optional config fetching.",
    solution_creators=["Kyle Harrington"],
    tags=["napari", "cellcanvas", "plugin", "visualization"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "type": "string",
            "default": "/Users/kharrington/Data/copick/cellcanvas_server/local_sshOverlay_localStatic.json",
            "description": "Path to the Copick configuration file.",
            "required": True
        }
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
