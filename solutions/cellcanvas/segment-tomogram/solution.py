###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """name: segmentation-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - scikit-learn==1.3.2
  - joblib
  - numpy
  - zarr
  - pip
"""

def run():
    import joblib
    import numpy as np
    import zarr

    def load_model(model_path):
        """Load the random forest model from a joblib file."""
        model = joblib.load(model_path)
        return model

    def apply_model_to_embeddings(embeddings_path, model, output_path):
        zarr_embeddings = zarr.open(embeddings_path, mode='r')
        # Assuming the output shape should match the transposed shape of the input embeddings
        output_shape = (zarr_embeddings.shape[2], zarr_embeddings.shape[1], zarr_embeddings.shape[0])
        output_zarr = zarr.open(output_path, shape=output_shape, dtype=int, mode='w', chunks=(200, 200, 200))

        # Process the entire array if chunking is not required
        embeddings = np.array(zarr_embeddings[:])
        embeddings_reshaped = embeddings.reshape(-1, embeddings.shape[-1])
        predictions = model.predict(embeddings_reshaped)
        predictions_reshaped = predictions.reshape(embeddings.shape[:-1])

        # Transpose the predictions to match the output shape
        predictions_transposed = np.transpose(predictions_reshaped, (2, 1, 0))
        output_zarr[:] = predictions_transposed

    embeddings_path = get_args().zarrembedding
    model_path = get_args().modelpath
    output_path = get_args().zarroutput

    model = load_model(model_path)
    apply_model_to_embeddings(embeddings_path, model, output_path)

    print(f"Segmentation output saved to {output_path}")


setup(
    group="cellcanvas",
    name="segment-tomogram",
    version="0.0.4",
    title="Segmentation using Random Forest on Embeddings in Chunks",
    description="Apply a Random Forest model to embeddings generated by TomoTwin to produce segmentation output, processing in chunks.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas team.", "url": "https://cellcanvas.org"}],
    tags=["segmentation", "random forest", "machine learning", "cryoet", "chunks"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "zarrembedding", "type": "string", "required": True, "description": "Path to the input Zarr file containing embeddings"},
        {"name": "zarroutput", "type": "string", "required": True, "description": "Path for the output Zarr file containing segmentation"},
        {"name": "modelpath", "type": "string", "required": True, "description": "Path to the joblib file containing the trained Random Forest model"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
