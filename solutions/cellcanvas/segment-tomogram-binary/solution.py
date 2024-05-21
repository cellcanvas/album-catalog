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
    voxel_spacing = int(args.voxel_spacing)
    run_name = args.run_name
    tomo_type = args.tomo_type
    feature_name = args.feature_name
    model_dir = args.model_dir
    segmentation_name = args.segmentation_name

    root = CopickRootFSSpec.from_file(copick_config_path)

    def get_prediction_segmentation(run, user_id, session_id, voxel_spacing, label):
        seg_name = f"{label}_{segmentation_name}"
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=False, name=seg_name, voxel_size=voxel_spacing)
        if not run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type):
            return None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, seg_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type):
                    return None
                shape = zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    def predict_segmentation(run, model_path, voxel_spacing, feature_name):
        features_list = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).features
        feature_path = None
        for feature in features_list:
            if feature.feature_type == feature_name:
                feature_path = feature.path
                break

        if not feature_path:
            raise ValueError(f"Feature with name '{feature_name}' not found in run '{run.name}'.")

        dataset_features = da.asarray(zarr.open(feature_path, "r"))
        chunk_shape = dataset_features.chunksize
        shape = dataset_features.shape
        
        prediction_data = np.zeros(shape[1:], dtype=np.uint16)
        
        model = joblib.load(model_path)

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
                    chunk_reshaped = chunk.transpose(1, 2, 3, 0).reshape(-1, chunk.shape[0])
                    predicted_chunk = model.predict(chunk_reshaped).reshape(chunk.shape[1:])
                    prediction_data[chunk_slice[1:]] = predicted_chunk
        
        return prediction_data

    run = root.get_run(run_name)
    if not run:
        raise ValueError(f"Run with name '{run_name}' not found.")

    label_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    for label_file in label_files:
        label = label_file.split('_')[0]
        model_path = os.path.join(model_dir, label_file)

        prediction_seg = get_prediction_segmentation(run, user_id, session_id, voxel_spacing, label)
        prediction_data = predict_segmentation(run, model_path, voxel_spacing, feature_name)

        prediction_seg[:] = prediction_data
        print(f"Prediction complete for label {label}. Segmentation saved as '{label}_{segmentation_name}'.")

setup(
    group="cellcanvas",
    name="segment-tomogram-binary",
    version="0.0.2",
    title="Predict Binary Segmentations Using Models",
    description="A solution that predicts binary segmentations for each label using models created by an optimization solution, and saves them separately.",
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
        {"name": "tomo_type", "type": "string", "required": True, "description": "Type of tomogram to use, e.g., denoised."},
        {"name": "feature_name", "type": "string", "required": True, "description": "Name of the feature to use, e.g., cellcanvas01."},
        {"name": "model_dir", "type": "string", "required": True, "description": "Directory containing the trained models."},
        {"name": "segmentation_name", "type": "string", "required": True, "description": "Name of the output segmentation."}        
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
