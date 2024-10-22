###album catalog: cellcanvas

from album.runner.api import setup, get_args
import subprocess

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - fastapi
  - uvicorn
  - pip
  - pip:
    - copick
    - album
"""

# Global status dictionary to store task status information
server_status = {
    "feature_generation": "not running",
    "model_training": "not running",
    "model_inference": "not running",
}

def run():
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import Optional
    from concurrent.futures import ThreadPoolExecutor
    import json
    import getpass
    import os
    import copick
    import logging
    import subprocess

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Global ThreadPoolExecutor for background tasks
    executor = ThreadPoolExecutor()

    args = get_args()
    app = FastAPI()

    allowed_solutions = {
        "cellcanvas:copick:generate-skimage-features",
        "cellcanvas:copick:generate-torch-basic-features",        
        "cellcanvas:copick:train-model-xgboost-copick",
        "cellcanvas:cellcanvas:segment-tomogram-xgboost"
    }

    # Dictionary for album solutions and versions
    album_solutions = {
        "generate_features": ("cellcanvas", "copick", "generate-torch-basic-features", "0.0.6"),
        "train_model": ("cellcanvas", "copick", "train-model-xgboost-copick", "0.0.4"),
        "run_model": ("cellcanvas", "cellcanvas", "segment-tomogram-xgboost", "0.0.7"),
    }

    copick_config_path = args.copick_config_path
    models_json_path = args.models_json_path
    current_username = getpass.getuser()

    print(f"Using config: {copick_config_path}")

    generated_models = {}
    if models_json_path and os.path.exists(models_json_path):
        with open(models_json_path, 'r') as f:
            generated_models = json.load(f)

    # Models for FastAPI endpoints
    class GenerateFeaturesArgs(BaseModel):
        copick_config_path: str
        run_name: str
        voxel_spacing: float
        tomo_type: str
        feature_type: str
        intensity: Optional[bool] = True
        edges: Optional[bool] = True
        texture: Optional[bool] = True
        sigma_min: Optional[float] = 0.5
        sigma_max: Optional[float] = 16.0

    class TrainModelArgs(BaseModel):
        copick_config_path: str
        painting_segmentation_names: str
        session_id: str
        user_id: str
        voxel_spacing: float
        tomo_type: str
        feature_types: str
        run_names: str
        eta: float = Field(default=0.3)
        gamma: float = Field(default=0.0)
        max_depth: int = Field(default=6)
        min_child_weight: float = Field(default=1.0)
        max_delta_step: float = Field(default=0.0)
        subsample: float = Field(default=1.0)
        colsample_bytree: float = Field(default=1.0)
        reg_lambda: float = Field(default=1.0)
        reg_alpha: float = Field(default=0.0)
        max_bin: int = Field(default=256)
        output_model_path: str

    class RunModelArgs(BaseModel):
        copick_config_path: str
        session_id: str
        user_id: str
        voxel_spacing: float
        run_name: str
        model_path: str
        tomo_type: str
        feature_names: str
        segmentation_name: str

    # Utility functions
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

    def run_album_solution_thread(catalog: str, group: str, name: str, version: str, args_list: list, task_type: str):
        args_str = " ".join(args_list)
        command = f"album run {catalog}:{group}:{name}:{version} {args_str}"

        logger.info(f"Running command: {command}")
        update_status(task_type, "running")

        try:
            proc = subprocess.run(command, shell=True, capture_output=True, text=True)
            stdout_str = proc.stdout
            stderr_str = proc.stderr

            logger.info(f"Command stdout: {stdout_str}")
            logger.error(f"Command stderr: {stderr_str}")

            if proc.returncode == 0:
                update_status(task_type, "completed")
                logger.info(f"Solution {name} completed successfully.")
            else:
                update_status(task_type, "error")
                logger.error(f"Solution {name} failed with return code {proc.returncode}.")
            return stdout_str, stderr_str
        except Exception as e:
            logger.exception(f"Error occurred while running the solution {name}: {str(e)}")
            update_status(task_type, "error")
            raise HTTPException(status_code=500, detail=str(e))

    # Install necessary album solutions
    def install_album_solutions():
        for solution_key, solution_details in album_solutions.items():
            catalog, group, name, version = solution_details
            solution = f"{catalog}:{group}:{name}:{version}"
            print(f"Installing solution: {solution}")
            install_command = f"album install {solution}"
            result = subprocess.run(install_command, shell=True, capture_output=True)
            if result.returncode != 0:
                print(f"Error installing {solution}: {result.stderr.decode()}")
            else:
                print(f"Successfully installed {solution}")

    # Install solutions when the server starts
    install_album_solutions()

    @app.post("/generate-features")
    async def generate_features_endpoint(solution_args: GenerateFeaturesArgs):
        logger.info(f"Received generate_features request: {solution_args}")
        catalog, group, name, version = album_solutions["generate_features"]
        check_solution_allowed(catalog, group, name)

        args_list = [
            "--copick_config_path", copick_config_path,
            "--run_name", solution_args.run_name,
            "--voxel_spacing", str(solution_args.voxel_spacing),
            "--tomo_type", solution_args.tomo_type,
            "--feature_type", solution_args.feature_type,
            "--intensity", str(solution_args.intensity),
            "--edges", str(solution_args.edges),
            "--texture", str(solution_args.texture),
            "--sigma_min", str(solution_args.sigma_min),
            "--sigma_max", str(solution_args.sigma_max),
            "--num_sigma", str(5)
        ]

        logger.info(f"Executing generate_features with args: {args_list}")
        executor.submit(run_album_solution_thread, catalog, group, name, version, args_list, "feature_generation")
        return {"message": "Feature generation started", "status": server_status["feature_generation"]}

    @app.post("/train-model")
    async def train_model_endpoint(solution_args: TrainModelArgs):
        logger.info(f"Received train_model request: {solution_args}")
        catalog, group, name, version = album_solutions["train_model"]
        check_solution_allowed(catalog, group, name)

        args_list = [
            "--copick_config_path", copick_config_path,
            "--painting_segmentation_names", solution_args.painting_segmentation_names,
            "--session_id", solution_args.session_id,
            "--user_id", solution_args.user_id,
            "--voxel_spacing", str(solution_args.voxel_spacing),
            "--tomo_type", solution_args.tomo_type,
            "--feature_types", solution_args.feature_types,
            "--run_names", solution_args.run_names,
            "--eta", str(solution_args.eta),
            "--gamma", str(solution_args.gamma),
            "--max_depth", str(solution_args.max_depth),
            "--min_child_weight", str(solution_args.min_child_weight),
            "--max_delta_step", str(solution_args.max_delta_step),
            "--subsample", str(solution_args.subsample),
            "--colsample_bytree", str(solution_args.colsample_bytree),
            "--reg_lambda", str(solution_args.reg_lambda),
            "--reg_alpha", str(solution_args.reg_alpha),
            "--max_bin", str(solution_args.max_bin),
            "--output_model_path", solution_args.output_model_path
        ]

        logger.info(f"Executing train_model with args: {args_list}")
        executor.submit(run_album_solution_thread, catalog, group, name, version, args_list, "model_training")
        return {"message": "Model training started", "status": server_status["model_training"]}

    @app.post("/run-model")
    async def run_model_endpoint(solution_args: RunModelArgs):
        logger.info(f"Received run_model request: {solution_args}")
        catalog, group, name, version = album_solutions["run_model"]
        check_solution_allowed(catalog, group, name)

        args_list = [
            "--copick_config_path", copick_config_path,
            "--model_path", solution_args.model_path,
            "--session_id", solution_args.session_id,
            "--user_id", solution_args.user_id,
            "--voxel_spacing", str(solution_args.voxel_spacing),
            "--run_name", solution_args.run_name,
            "--tomo_type", solution_args.tomo_type,
            "--feature_names", solution_args.feature_names,
            "--segmentation_name", solution_args.segmentation_name,
            "--write_mode", "immediate"
        ]

        logger.info(f"Executing run_model with args: {args_list}")
        executor.submit(run_album_solution_thread, catalog, group, name, version, args_list, "model_inference")
        return {"message": "Model inference started", "status": server_status["model_inference"]}

    @app.get("/models")
    def get_models():
        existing_models = {path: info for path, info in generated_models.items() if os.path.exists(path)}
        if len(existing_models) != len(generated_models):
            generated_models.clear()
            generated_models.update(existing_models)
            save_models_to_json()
        return {"models": existing_models}

    @app.post("/train-all")
    async def train_all(solution_args: TrainModelArgs):
        """Train the model across all available runs, generating features if necessary."""
        logger.info("Received train-all request")
        catalog, group, name, version = album_solutions["train_model"]
        feature_catalog, feature_group, feature_name, feature_version = album_solutions["generate_features"]
        check_solution_allowed(catalog, group, name)

        def train_all_task():
            run_names_to_train = []
            try:
                copick_project = copick.from_file(copick_config_path)

                for run in copick_project.runs:
                    logger.info(f"Checking run: {run.meta.name}")

                    voxel_spacing = run.get_voxel_spacing(solution_args.voxel_spacing)
                    if not voxel_spacing:
                        continue

                    tomo = voxel_spacing.get_tomogram(solution_args.tomo_type)

                    if len(tomo.features) > 0:
                        features = [f for f in tomo.features if f.feature_type == solution_args.feature_types]
                        if len(features) > 0:
                            features = features[0]
                    else:
                        features = None

                    if not features:
                        logger.info(f"Features not found for run {run.meta.name}, generating features...")

                        feature_args_list = [
                            "--copick_config_path", copick_config_path,
                            "--run_name", run.meta.name,
                            "--voxel_spacing", str(solution_args.voxel_spacing),
                            "--tomo_type", solution_args.tomo_type,
                            "--feature_type", solution_args.feature_types,
                            "--intensity", str(True),
                            "--edges", str(True),
                            "--texture", str(True),
                            "--sigma_min", str(0.5),
                            "--sigma_max", str(16.0),
                            "--num_sigma", str(5)
                        ]

                        logger.info(f"Executing feature generation for {run.meta.name}")
                        executor.submit(run_album_solution_thread, feature_catalog, feature_group, feature_name, feature_version, feature_args_list, "feature_generation").result()

                    if solution_args.painting_segmentation_names in run.segmentations:
                        run_names_to_train.append(run.meta.name)

                if run_names_to_train:
                    args_list = [
                        "--copick_config_path", copick_config_path,
                        "--painting_segmentation_names", solution_args.painting_segmentation_names,
                        "--session_id", solution_args.session_id,
                        "--user_id", solution_args.user_id,
                        "--voxel_spacing", str(solution_args.voxel_spacing),
                        "--tomo_type", solution_args.tomo_type,
                        "--feature_types", solution_args.feature_types,
                        "--run_names", ",".join(run_names_to_train),
                        "--eta", str(solution_args.eta),
                        "--gamma", str(solution_args.gamma),
                        "--max_depth", str(solution_args.max_depth),
                        "--min_child_weight", str(solution_args.min_child_weight),
                        "--max_delta_step", str(solution_args.max_delta_step),
                        "--subsample", str(solution_args.subsample),
                        "--colsample_bytree", str(solution_args.colsample_bytree),
                        "--reg_lambda", str(solution_args.reg_lambda),
                        "--reg_alpha", str(solution_args.reg_alpha),
                        "--max_bin", str(solution_args.max_bin),
                        "--output_model_path", solution_args.output_model_path
                    ]

                    logger.info(f"Executing train_all with args: {args_list}")
                    executor.submit(run_album_solution_thread, catalog, group, name, version, args_list, "model_training").result()
                else:
                    logger.info("No valid runs found for training")
            except Exception as e:
                logger.error(f"Error during train-all: {str(e)}")
                update_status("model_training", "error")

        # Submit the task to the executor
        executor.submit(train_all_task)
        return {"message": "Train-all task submitted", "status": server_status["model_training"]}

    @app.post("/predict-all")
    async def predict_all(solution_args: RunModelArgs):
        """Predict using the model across all available runs. Computes features first if necessary."""
        logger.info(f"Received predict-all request: {solution_args}")
        catalog, group, name, version = album_solutions["run_model"]
        feature_catalog, feature_group, feature_name, feature_version = album_solutions["generate_features"]
        check_solution_allowed(catalog, group, name)

        def predict_all_task():
            try:
                copick_project = copick.from_file(copick_config_path)
                for run in copick_project.runs:
                    logger.info(f"Processing run: {run.meta.name}")

                    feature_args_list = [
                        "--copick_config_path", copick_config_path,
                        "--run_name", run.meta.name,
                        "--voxel_spacing", str(solution_args.voxel_spacing),
                        "--tomo_type", solution_args.tomo_type,
                        "--feature_type", solution_args.feature_names,
                        "--intensity", str(True),
                        "--edges", str(True),
                        "--texture", str(True),
                        "--sigma_min", str(0.5),
                        "--sigma_max", str(16.0),
                        "--num_sigma", str(5)
                    ]

                    logger.info(f"Executing feature generation for {run.meta.name}")
                    executor.submit(run_album_solution_thread, feature_catalog, feature_group, feature_name, feature_version, feature_args_list, "feature_generation").result()

                    args_list = [
                        "--copick_config_path", copick_config_path,
                        "--model_path", solution_args.model_path,
                        "--session_id", solution_args.session_id,
                        "--user_id", solution_args.user_id,
                        "--voxel_spacing", str(solution_args.voxel_spacing),
                        "--run_name", run.meta.name,
                        "--tomo_type", solution_args.tomo_type,
                        "--feature_names", solution_args.feature_names,
                        "--segmentation_name", solution_args.segmentation_name,
                        "--write_mode", "immediate"
                    ]

                    logger.info(f"Executing prediction for {run.meta.name}")
                    executor.submit(run_album_solution_thread, catalog, group, name, version, args_list, "model_inference").result()

            except Exception as e:
                logger.error(f"Error during predict-all: {str(e)}")
                update_status("model_inference", "error")

        # Submit the task to the executor
        executor.submit(predict_all_task)
        return {"message": "Predict-all task submitted", "status": server_status["model_inference"]}


    @app.get("/status")
    async def get_status():
        logger.info("Received status request")
        logger.info(f"Current server status: {server_status}")
        return server_status

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

setup(
    group="cellcanvas",
    name="experimental-server",
    version="0.0.17",
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
