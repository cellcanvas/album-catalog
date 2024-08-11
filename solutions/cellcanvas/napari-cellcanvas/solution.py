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
  - pip:
    - album
    - "copick[all]"
    - "git+https://github.com/kephale/napari-cellcanvas.git"
"""

def run():
    from napari import Viewer
    from napari_cellcanvas import CellCanvasWidget
    import napari

    # Retrieve the arguments passed to the solution
    args = get_args()

    copick_config_path = args.copick_config_path
    hostname = args.hostname
    port = args.port

    # Initialize the Napari viewer
    viewer = Viewer()

    # Initialize the CellCanvasWidget with the provided arguments
    widget = CellCanvasWidget(viewer=viewer, copick_config_path=copick_config_path, hostname=hostname, port=port)

    # Add the widget to the viewer's dock
    viewer.window.add_dock_widget(widget, area='right')

    # Start Napari
    napari.run()

setup(
    group="cellcanvas",
    name="napari-cellcanvas",
    version="0.0.4",
    title="napari-CellCanvas",
    description="A solution launches napari-cellcanvas.",
    solution_creators=["Kyle Harrington"],
    tags=["napari", "cellcanvas", "plugin", "visualization"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "type": "string",
            "default": "/Users/kharrington/Data/copick/cellcanvas_server/local_sshOverlay_localStatic.json",
            "description": "Path to the Copick configuration file."
        },
        {
            "name": "hostname",
            "type": "string",
            "default": "localhost",
            "description": "Hostname for the server."
        },
        {
            "name": "port",
            "type": "integer",
            "default": 8080,
            "description": "Port number for the server."
        }
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
