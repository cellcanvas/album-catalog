###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - copick
"""

def run():
    import joblib
    import numpy as np
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec
    import os
    import dask.array as da

    args = get_args()
    copick_config_path = args.copick_config_path
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = args.voxel_spacing
    run_name = args.run_name
    model_path = args.model_path

    # Load the Copick root from the configuration file
    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_prediction_segmentation(run, user_id, session_id, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name="predictionsegmentation", voxel_size=voxel_spacing)
        if not run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised"):
            return None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, "predictionsegmentation", session_id, True, user_id=user_id
            )
            shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised"):
                    return None
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    # Function to predict segmentation
    def predict_segmentation(run, model_path, voxel_spacing):
        dataset_features = da.asarray(zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised").features[0].path, "r"))
        chunk_shape = dataset_features.chunksize
        shape = dataset_features.shape
        
        # Placeholder for the prediction data
        prediction_data = np.zeros(shape[1:], dtype=np.uint16)
        
        # Load the model
        model = joblib.load(model_path)

        # Iterate over chunks and predict
        for z in range(0, shape[1], chunk_shape[1]):
            for y in range(0, shape[2], chunk_shape[2]):
                for x in range(0, shape[3], chunk_shape[3]):
                    chunk_slice = (
                        slice(None),
                        slice(z, min(z + chunk_shape[1], shape[1])),
                        slice(y, min(y + chunk_shape[2], shape[2])),
                        slice(x, min(x + chunk_shape[3], shape[3]))
                    )
                    chunk = dataset_features[chunk_slice].compute()
                    chunk_reshaped = chunk.reshape(chunk.shape[1] * chunk.shape[2] * chunk.shape[3], chunk.shape[0])
                    predicted_chunk = model.predict(chunk_reshaped).reshape(chunk.shape[1:])
                    prediction_data[chunk_slice[1:]] = predicted_chunk
        
        return prediction_data

    # Retrieve the specified run by name
    run = root.get_run(run_name)
    if not run:
        raise ValueError(f"Run with name '{run_name}' not found.")

    print(f"Predicting run '{run_name}': {run}")

    prediction_seg = get_prediction_segmentation(run, user_id, session_id, voxel_spacing)
    prediction_data = predict_segmentation(run, model_path, voxel_spacing)

    # Save the prediction data to the Zarr array
    prediction_seg[:] = prediction_data

    print("Prediction complete. Segmentation saved as 'predictionsegmentation'.")

setup(
    group="cellcanvas",
    name="segment-tomogram",
    version="0.1.3",
    title="Predict Segmentation Using a Model",
    description="A solution that predicts segmentation using a model for a Copick project and saves it as 'predictionsegmentation'.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "segmentation", "prediction", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "model_path", "type": "string", "required": True, "description": "Path to the trained model file."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
