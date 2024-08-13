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
    - "git+https://github.com/kephale/napari-cellcanvas.git"
    - "sshfs>=2024.6.0"
"""

def run():
    from napari import Viewer
    from napari_cellcanvas import CellCanvasWidget
    import napari
    import requests
    import json
    import os

    # Retrieve the arguments passed to the solution
    args = get_args()

    copick_config_path = args.copick_config_path
    hostname = args.hostname
    port = args.port
    fetch_config = args.fetch_config
    overlay_remote = args.overlay_remote
    static_remote = args.static_remote
    overlay_path = args.overlay_path
    static_path = args.static_path

    if fetch_config:
        # Construct the URL for the API request
        url = f"http://{hostname}:{port}/get-copick-config"
        
        # Prepare the parameters for the API request
        params = {
            "overlay_remote": overlay_remote,
            "static_remote": static_remote
        }

        # Send the GET request to the server to fetch the config
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch config: {response.text}")

        config = response.json()["config"]

        # If paths are not remote, insert the local paths provided
        if not overlay_remote:
            config["overlay_root"] = overlay_path or config["overlay_root"]
        if not static_remote:
            config["static_root"] = static_path or config["static_root"]

        # Write the final configuration to the specified file
        os.makedirs(os.path.dirname(copick_config_path), exist_ok=True)
        with open(copick_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Configuration fetched and written to {copick_config_path}")

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
    version="0.0.5",
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
        },
        {
            "name": "hostname",
            "type": "string",
            "default": "localhost",
            "description": "Hostname for the server.",
            "required": True
        },
        {
            "name": "port",
            "type": "integer",
            "default": 8080,
            "description": "Port number for the server.",
            "required": True
        },
        {
            "name": "fetch_config",
            "type": "boolean",
            "default": False,
            "description": "Whether to fetch the config from the server.",
            "required": False
        },
        {
            "name": "overlay_remote",
            "type": "boolean",
            "default": False,
            "description": "Set to true if the overlay path should be remote (SSH).",
            "required": False
        },
        {
            "name": "static_remote",
            "type": "boolean",
            "default": False,
            "description": "Set to true if the static path should be remote (SSH).",
            "required": False
        },
        {
            "name": "overlay_path",
            "type": "string",
            "description": "The local path for the overlay root if not remote.",
            "required": False
        },
        {
            "name": "static_path",
            "type": "string",
            "description": "The local path for the static root if not remote.",
            "required": False
        }
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
