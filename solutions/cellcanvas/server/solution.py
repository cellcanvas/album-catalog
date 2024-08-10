###album catalog: cellcanvas

from album.runner.api import setup

env_file = """
channels:
  - conda-forge
  - defaults
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
    from typing import Optional, Dict, Any
    from album.api import Album
    from album.core.utils.operations.solution_operations import (
        get_deploy_dict,
    )

    import io
    import json
    from contextlib import redirect_stdout, redirect_stderr

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
        "cellcanvas:cellcanvas:train-model-xgboost",
        "cellcanvas:cellcanvas:segment-tomogram-xgboost",
        "cellcanvas:morphospaces:train_swin_unetr_pixel_embedding",
        "cellcanvas:copick:submit-album-job-array"
    }

    class SolutionArgs(BaseModel):
        args: Optional[Dict[str, Any]] = {}

    def check_solution_allowed(catalog: str, group: str, name: str):
        solution_path = f"{catalog}:{group}:{name}"
        if solution_path not in allowed_solutions:
            raise HTTPException(status_code=403, detail="Solution not allowed")

    @app.post("/run/{catalog}/{group}/{name}/{version}")
    def run_solution_endpoint(catalog: str, group: str, name: str, version: str, solution_args: SolutionArgs):
        check_solution_allowed(catalog, group, name)
        try:
            args_list = []
            for key, value in solution_args.args.items():
                args_list.extend([f"--{key} {str(value)}"])

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
    version="0.0.2",
    title="FastAPI Album Server",
    description="A FastAPI server to manage Album solutions.",
    solution_creators=["Kyle Harrington"],
    tags=["fastapi", "album", "server"],
    license="MIT",
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
