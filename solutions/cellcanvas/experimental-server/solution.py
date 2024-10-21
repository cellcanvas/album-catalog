###album catalog: cellcanvas

from album.runner.api import setup, get_args
import asyncio
import subprocess

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - fastapi
  - uvicorn
  - album
  - pip
  - pip:
    - copick
"""

# Global status dictionary to store task status information
server_status = {
    "feature_generation": "not running",
    "model_training": "not running",
    "model_inference": "not running",
}

def run():
    import sys
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional, Dict, Any
    from album.api import Album
    from album.core.utils.operations.solution_operations import (
        get_deploy_dict,
    )
    import io
    import json
    import getpass
    import os
    import copick

    args = get_args()
    app = FastAPI()

    album_instance = Album.Builder().build()
    album_instance.load_or_create_collection()

    allowed_solutions = {
        "cellcanvas:copick:generate-skimage-features",
        "cellcanvas:copick:generate-torch-basic-features",        
        "cellcanvas:copick:train-model-xgboost-copick",
        "cellcanvas:cellcanvas:segment-tomogram-xgboost"
    }

    copick_config_path = args.copick_config_path
    models_json_path = args.models_json_path
    current_username = getpass.getuser()

    print(f"Using config: {copick_config_path}")

    generated_models = {}
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

    def update_status(task_type, status):
        server_status[task_type] = status

    async def run_album_solution_async(catalog: str, group: str, name: str, version: str, args_list: list):
        """Runs an album solution asynchronously using subprocess."""
        args_str = " ".join(args_list)
        update_status(name, "running")
        try:
            command = f"album run {catalog}:{group}:{name}:{version} {args_str}"
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                update_status(name, "completed")
            else:
                update_status(name, "error")
            return stdout.decode(), stderr.decode()
        except Exception as e:
            update_status(name, "error")
            return str(e), ""

    @app.post("/generate-features")
    async def generate_features_endpoint(solution_args: SolutionArgs):
        try:
            catalog, group, name, version = "cellcanvas", "copick", "generate-torch-basic-features", "0.0.3"
            check_solution_allowed(catalog, group, name)

            args_list = []
            for key, value in solution_args.args.items():
                args_list.extend([f"--{key}", f"{str(value)}"])

            args_list.extend(["--copick_config_path", copick_config_path])

            stdout, stderr = await run_album_solution_async(catalog, group, name, version, args_list)
            return {"stdout": stdout, "stderr": stderr}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/train-model")
    async def train_model_endpoint(solution_args: SolutionArgs):
        try:
            catalog, group, name, version = "cellcanvas", "copick", "train-model-xgboost-copick", "0.0.1"
            check_solution_allowed(catalog, group, name)

            args_list = []
            for key, value in solution_args.args.items():
                args_list.extend([f"--{key}", f"{str(value)}"])

            args_list.extend(["--copick_config_path", copick_config_path])

            stdout, stderr = await run_album_solution_async(catalog, group, name, version, args_list)
            return {"stdout": stdout, "stderr": stderr}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/run-model")
    async def run_model_endpoint(solution_args: SolutionArgs):
        try:
            catalog, group, name, version = "cellcanvas", "cellcanvas", "segment-tomogram-xgboost", "0.0.5"
            check_solution_allowed(catalog, group, name)

            args_list = []
            for key, value in solution_args.args.items():
                args_list.extend([f"--{key}", f"{str(value)}"])

            args_list.extend(["--copick_config_path", copick_config_path])

            stdout, stderr = await run_album_solution_async(catalog, group, name, version, args_list)
            return {"stdout": stdout, "stderr": stderr}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/models")
    def get_models():
        try:
            existing_models = {path: info for path, info in generated_models.items() if os.path.exists(path)}
            if len(existing_models) != len(generated_models):
                generated_models.clear()
                generated_models.update(existing_models)
                save_models_to_json()
            return {"models": existing_models}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

    @app.get("/status")
    def get_status(dataset: str):
        try:
            status_info = {"runs": 0, "annotations_exist": False, "features_exist": False, "predictions_exist": False}

            # Check if the dataset exists in the copick config
            if os.path.exists(copick_config_path):
                root = copick.from_file(copick_config_path)
                runs = root.runs
                status_info["runs"] = len(runs)
                status_info["run_id"] = dataset

                if runs:
                    # Check for features, annotations, and predictions for the selected dataset
                    for run in runs:
                        if hasattr(run, 'features') and run.features:
                            status_info["features_exist"] = True
                        if hasattr(run, 'annotations') and run.annotations:
                            status_info["annotations_exist"] = True
                        if hasattr(run, 'predictions') and run.predictions:
                            status_info["predictions_exist"] = True

            # Add the current status for features/model training/inference
            status_info["feature_generation"] = server_status["feature_generation"]
            status_info["model_training"] = server_status["model_training"]
            status_info["model_inference"] = server_status["model_inference"]

            return status_info
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error occurred while fetching status: {error_trace}")
            raise HTTPException(status_code=500, detail=f"Error fetching status: {str(e)}")


    uvicorn.run(app, host="0.0.0.0", port=8000)

setup(
    group="cellcanvas",
    name="experimental-server",
    version="0.0.4",
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