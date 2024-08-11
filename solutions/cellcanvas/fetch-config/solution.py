###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - requests
"""

def run():
    import requests
    import json

    # Get the arguments passed to the solution
    args = get_args()
    
    localhost = args.localhost
    port = args.port
    overlay_remote = args.overlay_remote
    static_remote = args.static_remote
    overlay_path = args.overlay_path
    static_path = args.static_path
    filepath = args.filepath

    # Construct the URL for the API request
    url = f"http://{localhost}:{port}/get-copick-config"
    
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
        config["overlay_root"] = overlay_path
    if not static_remote:
        config["static_root"] = static_path

    # Write the final configuration to the specified file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration written to {filepath}")

setup(
    group="cellcanvas",
    name="fetch-config",
    version="0.0.1",
    title="Fetch Copick Config and Write to File",
    description="Fetches a Copick config from a FastAPI server and writes it to a file.",
    solution_creators=["Kyle Harrington"],
    tags=["fastapi", "album", "config", "fetch"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "localhost",
            "type": "string",
            "default": "127.0.0.1",
            "description": "The localhost where the FastAPI server is running."
        },
        {
            "name": "port",
            "type": "integer",
            "default": 8000,
            "description": "The port on which the FastAPI server is listening."
        },
        {
            "name": "overlay_remote",
            "type": "boolean",
            "default": False,
            "description": "Set to true if the overlay path should be remote (SSH)."
        },
        {
            "name": "static_remote",
            "type": "boolean",
            "default": False,
            "description": "Set to true if the static path should be remote (SSH)."
        },
        {
            "name": "overlay_path",
            "type": "string",
            "default": "local:///path/to/overlay",
            "description": "The local path for the overlay root if not remote."
        },
        {
            "name": "static_path",
            "type": "string",
            "default": "local:///path/to/static",
            "description": "The local path for the static root if not remote."
        },
        {
            "name": "filepath",
            "type": "string",
            "default": "/path/to/output/config.json",
            "description": "The file path where the config will be written."
        }
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
