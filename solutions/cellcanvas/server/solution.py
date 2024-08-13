###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - fastapi
  - uvicorn
  - album
"""

def run():
    import sys
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional, Dict, Any, List
    from album.api import Album
    from album.core.utils.operations.solution_operations import (
        get_deploy_dict,
    )
    import io
    import json
    import getpass
    from contextlib import redirect_stdout, redirect_stderr
    import os

    args = get_args()
    
    app = FastAPI()

    # Initialize the Album instance outside the function to avoid issues with scope
    print("Initializing Album instance...")
    album_instance = Album.Builder().build()
    album_instance.load_or_create_collection()
    print("Album instance initialized.")

    allowed_solutions = {
        "cellcanvas:cellcanvas:generate-pixel-embedding",
        "cellcanvas:polnet:generate-tomogram",
        "cellcanvas:copick:paint-from-picks",
        "cellcanvas:copick:generate-skimage-features",
        "cellcanvas:copick:generate-torch-basic-features",        
        # "cellcanvas:cellcanvas:train-model-xgboost",
        "cellcanvas:cellcanvas:segment-tomogram-xgboost",
        "cellcanvas:morphospaces:train_swin_unetr_pixel_embedding",
        "cellcanvas:copick:submit-album-job-array",
        "cellcanvas:copick:train-model-xgboost-copick"
    }

    copick_config_path = args.copick_config_path
    models_json_path = args.models_json_path

    # Get the username of the user running the server
    current_username = getpass.getuser()

    # Dictionary to store information about generated models
    generated_models = {}

    # Load existing models from JSON if the file exists
    if models_json_path and os.path.exists(models_json_path):
        with open(models_json_path, 'r') as f:
            generated_models = json.load(f)
    
    class SolutionArgs(BaseModel):
        args: Optional[Dict[str, Any]] = {}

    def save_models_to_json():
        if models_json_path:
            with open(models_json_path, 'w') as f:
                json.dump(generated_models, f, indent=2)
        
    def check_solution_allowed(catalog: str, group: str, name: str):
        solution_path = f"{catalog}:{group}:{name}"
        if solution_path not in allowed_solutions:
            raise HTTPException(status_code=403, detail="Solution not allowed")

    @app.post("/run/{catalog}/{group}/{name}/{version}")
    def run_solution_endpoint(catalog: str, group: str, name: str, version: str, solution_args: SolutionArgs):
        check_solution_allowed(catalog, group, name)
        try:
            args_list = [""]
            for key, value in solution_args.args.items():
                args_list.extend([f"--{key}"])
                args_list.extend([f"{str(value)}"])

            # Add copick_config_path to the arguments
            args_list.extend(["--copick_config_path", copick_config_path])

            # Log the constructed argument list for debugging
            print(f"Running solution with arguments: {args_list}")

            # Capture stdout and stderr
            output = io.StringIO()
            error_output = io.StringIO()

            try:
                with redirect_stdout(output), redirect_stderr(error_output):
                    result = album_instance.run(f"{catalog}:{group}:{name}:{version}", args_list)

                # Log output and errors
                print(f"Task completed successfully. Output: {output.getvalue()}")
                if error_output.getvalue():
                    print(f"Errors: {error_output.getvalue()}")

                # Check if this solution generates a model
                output_model_path = next((arg.split('=')[1] for arg in args_list if arg.startswith('--output_model_path=')), None)
                if output_model_path:
                    generated_models[output_model_path] = {
                        "catalog": catalog,
                        "group": group,
                        "name": name,
                        "version": version
                    }
                    save_models_to_json()
                    
                return {
                    "result": result,
                    "stdout": output.getvalue(),
                    "stderr": error_output.getvalue()
                }
            except Exception as e:
                print(f"Task failed with error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        except Exception as e:
            # Log the full error
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error occurred while running solution: {error_trace}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    @app.get("/models")
    def get_models():
        try:
            # Filter out models that no longer exist on the filesystem
            existing_models = {path: info for path, info in generated_models.items() if os.path.exists(path)}
            
            # Update the JSON file if any models were removed
            if len(existing_models) != len(generated_models):
                generated_models.clear()
                generated_models.update(existing_models)
                save_models_to_json()
            
            return {"models": existing_models}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

    @app.get("/user_session_ids")
    def get_user_session_ids():
        try:
            with open(copick_config_path, 'r') as config_file:
                config_data = json.load(config_file)
            
            user_ids = set()
            session_ids = set()

            # Extract user_ids and session_ids from the config
            for run in config_data.get('runs', []):
                user_ids.add(run.get('user_id', ''))
                session_ids.add(str(run.get('session_id', '')))
            
            for pick in config_data.get('picks', []):
                user_ids.add(pick.get('user_id', ''))
                session_ids.add(str(pick.get('session_id', '')))

            return {
                "user_ids": list(user_ids),
                "session_ids": list(session_ids)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching user and session IDs: {str(e)}")

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    @app.post("/install/{catalog}/{group}/{name}/{version}")
    def install_solution_endpoint(catalog: str, group: str, name: str, version: str):
        check_solution_allowed(catalog, group, name)
        try:
            result = album_instance.install(f"{catalog}:{group}:{name}:{version}")
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/uninstall/{catalog}/{group}/{name}/{version}")
    def uninstall_solution_endpoint(catalog: str, group: str, name: str, version: str):
        check_solution_allowed(catalog, group, name)
        try:
            result = album_instance.uninstall(f"{catalog}:{group}:{name}:{version}")
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/info/{catalog}/{group}/{name}/{version}")
    def info_solution_endpoint(catalog: str, group: str, name: str, version: str):
        check_solution_allowed(catalog, group, name)
        try:
            # Construct the solution path (assuming standard format)
            solution_path = f"{catalog}:{group}:{name}:{version}"

            # Resolve the solution using the Album instance
            resolve_result = album_instance.resolve(solution_path)

            # Load the solution
            solution = resolve_result.loaded_solution()

            # Convert the solution to a deployable dictionary or string
            deploy_dict = get_deploy_dict(solution)

            # Return the solution information
            return {"info": deploy_dict}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/index")
    def index_endpoint():
        try:
            index = album_instance.get_index_as_dict()
            filtered_index = {}

            if isinstance(index, dict) and 'catalogs' in index:
                for catalog in index['catalogs']:
                    catalog_name = "cellcanvas"  # Replace with your actual catalog name if needed
                    if isinstance(catalog, dict) and 'solutions' in catalog:
                        for solution in catalog['solutions']:
                            if isinstance(solution, dict) and 'setup' in solution:
                                setup = solution['setup']
                                solution_path = f"{catalog_name}:{setup.get('group', '')}:{setup.get('name', '')}"

                                if solution_path in allowed_solutions:
                                    key = f"{catalog['name']}:{setup.get('group')}:{setup.get('name')}"
                                    # Explicitly add the catalog name to the solution info
                                    setup['catalog'] = catalog_name
                                    filtered_index[key] = setup  # Store the setup dict directly

                print(f"Number of items in filtered index: {len(filtered_index)}")
            else:
                raise HTTPException(status_code=500, detail="Index structure is not as expected")

            return {"index": filtered_index}
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error occurred while fetching index: {error_trace}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    @app.get("/get-copick-config")
    def get_copick_config(overlay_remote: bool = False, static_remote: bool = False):
        try:
            with open(copick_config_path, 'r') as config_file:
                config_data = json.load(config_file)

            # Modify paths based on whether they are remote or local
            if overlay_remote:
                config_data['overlay_root'] = config_data['overlay_root'].replace("local://", "ssh://")
                config_data['overlay_fs_args'] = {
                    "username": current_username,
                    "host": "localhost",
                    "port": 2222
                }
            else:
                config_data['overlay_fs_args'] = {
                    "auto_mkdir": True
                }

            if static_remote:
                config_data['static_root'] = config_data['static_root'].replace("local://", "ssh://")
                config_data['static_fs_args'] = {
                    "username": current_username,
                    "host": "localhost",
                    "port": 2222
                }
            else:
                config_data['static_fs_args'] = {
                    "auto_mkdir": True
                }

            # Include placeholders for paths if necessary
            if not overlay_remote:
                config_data['overlay_root'] = "{overlay_root}"
            if not static_remote:
                config_data['static_root'] = "{static_root}"

            return {"config": config_data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/update")
    def update_endpoint():
        try:
            result = album_instance.update()
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/add-catalog")
    def add_catalog_endpoint(catalog_url: str):
        try:
            result = album_instance.add_catalog(catalog_url)
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/upgrade")
    def upgrade_endpoint():
        try:
            result = album_instance.upgrade()
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

setup(
    group="cellcanvas",
    name="server",
    version="0.0.11",
    title="FastAPI CellCanvas Server",
    description="Backend for CellCanvas with Copick Config Support.",
    solution_creators=["Kyle Harrington"],
    tags=["fastapi", "album", "server", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "type": "string",
            "default": "/path/to/copick/config.json",
            "description": "Path to the Copick configuration file."
        },
        {
            "name": "models_json_path",
            "type": "string",
            "default": "",
            "description": "Path to the JSON file for storing model listings. If not provided, model listings will not be persisted.",
            "required": False
        }        
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
