###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - s3fs
  - matplotlib
  - pip:
    - album
    - cryoet-data-portal
    - copick
    - ndjson
    - tenacity    
"""

def generate_unique_colors(n):
    """Generate a list of n visually distinct colors in RGBA format."""
    import matplotlib.colors as mcolors
    import numpy as np
    
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb([hue, 0.8, 0.9]) for hue in hues]
    rgba_colors = [[int(c[0]*255), int(c[1]*255), int(c[2]*255), 128] for c in colors]
    return rgba_colors

def run():
    import os
    import json
    import s3fs
    import zarr
    import ndjson
    from zarr.storage import KVStore
    from cryoet_data_portal import Client, Run, AnnotationFile, Tomogram, TiltSeries
    import copick
    from copick.models import CopickPoint, CopickConfig
    from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s ~ %(message)s')
    logger = logging.getLogger(__name__)

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
    def safe_query_run(annotation, run):
        try:
            if annotation.annotation.tomogram_voxel_spacing.run_id != run.id:
                return False
            return True
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise e

    args = get_args()
    dataset_id = args.dataset_id
    copick_config_path = args.copick_config_path
    voxel_spacing_input = float(args.voxel_spacing) if args.voxel_spacing else None
    overlay_root = args.overlay_root or os.path.expanduser("~/copick_overlay")
    static_root = args.static_root or os.path.expanduser("~/copick_static")

    fs = s3fs.S3FileSystem(anon=True)
    data_path = get_data_path()

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    client = Client()
    
    runs = Run.find(client, [Run.dataset_id == int(dataset_id)])
    if not runs:
        raise ValueError("No runs found for the given dataset ID.")

    annotations = AnnotationFile.find(client, [
        AnnotationFile.annotation.tomogram_voxel_spacing.run.dataset_id == int(dataset_id)
    ])

    max_colors = 32
    distinct_colors = generate_unique_colors(max_colors)

    pickable_objects = {}
    for annotation in annotations:
        object_name = annotation.annotation.object_name
        if object_name not in pickable_objects:
            label_index = len(pickable_objects)
            color = distinct_colors[label_index % max_colors]
            pickable_objects[object_name] = {
                "name": object_name,
                "is_particle": annotation.shape_type in ["Point", "OrientedPoint"],
                "label": label_index + 1,
                "color": color,
                "go_id": annotation.annotation.object_id
            }

    copick_config = {
        "name": "auto_generated_config",
        "description": "Auto-generated config from CryoET data portal",
        "version": "0.1.5",
        "user_id": "albumImportFromCryoETDataPortal",
        "pickable_objects": list(pickable_objects.values()),
        "overlay_root": f"local://{overlay_root}",
        "static_root": f"local://{static_root}",
        "overlay_fs_args": {"auto_mkdir": True},
        "static_fs_args": {"auto_mkdir": True}
    }

    with open(copick_config_path, 'w') as f:
        json.dump(copick_config, f, indent=4)

    root = copick.from_file(copick_config_path)
    
    for run in runs:
        run_name = run.name
        logger.info(f"Processing run: {run_name}")

        copick_run = root.get_run(run_name)
        if not copick_run:
            copick_run = root.new_run(run_name)

        for tvs in run.tomogram_voxel_spacings:
            voxel_spacing_value = tvs.voxel_spacing
            if voxel_spacing_input is not None and voxel_spacing_value != voxel_spacing_input:
                continue
            
            if voxel_spacing_value not in [vs.voxel_size for vs in copick_run.voxel_spacings]:
                voxel_spacing = copick_run.new_voxel_spacing(voxel_size=voxel_spacing_value)
            else:
                voxel_spacing = copick_run.get_voxel_spacing(voxel_spacing_value)

            for tomo in tvs.tomograms:
                s3_zarr_path = tomo.s3_omezarr_dir
                
                tomogram_name = f"{tomo.name}"
                logger.info(f"Adding tomogram to Copick: {tomogram_name}")

                copick_tomogram = voxel_spacing.new_tomogram(tomogram_name)
                
                try:
                    logger.info(f"Opening S3 store: {s3_zarr_path}")
                    s3_store = KVStore(zarr.storage.FSStore(f's3://{s3_zarr_path}', fs=fs))
                    
                    logger.info(f"Opening Copick Zarr store for tomogram: {tomogram_name}")
                    copick_store = KVStore(copick_tomogram.zarr())
                    
                    logger.info(f"Streaming data from {s3_zarr_path} to Copick Zarr store for tomogram {tomogram_name}")
                    copick_store_group = zarr.group(store=copick_store)

                    s3_store_keys = list(s3_store.keys())
                    logger.info(f"S3 Store Content Keys: {s3_store_keys}")
                    
                    for key in s3_store_keys:
                        logger.info(f"Copying dataset {key} from S3 to Copick store.")
                        zarr.copy_all(zarr.open_group(s3_store, mode='r'), copick_store_group)
                except Exception as e:
                    logger.error(f"Error during Zarr copy for tomogram {tomogram_name}: {e}")
                    continue

        # Process annotations (including segmentations)
        for annotation in annotations:
            try:
                if not safe_query_run(annotation, run):
                    continue
            except RetryError:
                logger.error(f"Failed after multiple retries for annotation {annotation.annotation.object_name}")
                continue

            object_name = annotation.annotation.object_name
            label_num = pickable_objects[object_name]["label"]
            user_id = "albumImportFromCryoETDataPortal"
            session_id = "0"
            logger.info(f"Processing annotation for object: {object_name}, label: {label_num}")

            if annotation.shape_type in ["Point", "OrientedPoint"]:
                with fs.open(annotation.s3_path) as pointfile:
                    points = list(ndjson.reader(pointfile))
                    if not points:
                        continue
                    
                    centroids = [(p['location']['z'], p['location']['y'], p['location']['x']) for p in points]
                    logger.info(f"Saving {len(centroids)} picks for label {label_num}")
                    pick_set = copick_run.new_picks(object_name, session_id, user_id)
                    pick_set.points = [CopickPoint(location={'x': c[2] * voxel_spacing_value, 'y': c[1] * voxel_spacing_value, 'z': c[0] * voxel_spacing_value}) for c in centroids]
                    pick_set.store()
            
            elif annotation.shape_type == "SegmentationMask" and annotation.format == "zarr":
                logger.info(f"Processing segmentation for object: {object_name}")
                seg_zarr_path = annotation.s3_path
                
                copick_segmentation = copick_run.new_segmentation(
                    voxel_size=voxel_spacing_value,
                    name=object_name,
                    session_id=session_id,
                    is_multilabel=False,
                    user_id=user_id
                )
                
                try:
                    s3_seg_store = KVStore(zarr.storage.FSStore(f's3://{seg_zarr_path}', fs=fs))
                    copick_seg_store = KVStore(copick_segmentation.zarr())
                    
                    logger.info(f"Streaming segmentation data from {seg_zarr_path} to Copick Zarr store")
                    copick_seg_group = zarr.group(store=copick_seg_store)
                    
                    zarr.copy_all(zarr.open_group(s3_seg_store, mode='r'), copick_seg_group)
                except Exception as e:
                    logger.error(f"Error during segmentation Zarr copy for object {object_name}: {e}")
                    continue

    # Update Copick configuration with the correct paths
    copick_config.overlay_root = f"local://{overlay_root}"
    copick_config.static_root = f"local://{static_root}"
    with open(copick_config_path, 'w') as f:
        json.dump(copick_config.model_dump(exclude_unset=True), f, indent=4)

    logger.info(f"Done with all runs. Outputs are saved in the Copick project at {copick_config_path}")


setup(
    group="copick",
    name="project_from_dataportal",
    version="0.2.2",
    title="Fetch Data from CryoET Data Portal and Integrate with Copick",
    description="Fetches tomograms, tilt series, annotations, segmentations, and metadata from cryoet_data_portal and integrates them into the specified Copick project.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Cellcanvas and Copick teams", "url": "https://cellcanvas.org"}],
    tags=["zarr", "deep learning", "tomography", "cryoET"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "dataset_id", "type": "string", "required": True, "description": "Dataset ID to process."},
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration file."},
        {"name": "voxel_spacing", "type": "string", "required": False, "description": "Optional voxel spacing to filter tomograms."},
        {"name": "overlay_root", "type": "string", "required": False, "description": "Path to the overlay root directory."},
        {"name": "static_root", "type": "string", "required": False, "description": "Path to the static root directory."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)