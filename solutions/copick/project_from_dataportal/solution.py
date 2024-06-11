###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
  - defaults
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
"""

def generate_unique_colors(n):
    """Generate a list of n visually distinct colors in RGBA format."""
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Generate n equally spaced hues
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb([hue, 0.8, 0.9]) for hue in hues]
    
    # Convert to RGBA with a set alpha value
    rgba_colors = [[int(c[0]*255), int(c[1]*255), int(c[2]*255), 128] for c in colors]
    return rgba_colors

def run():
    import os
    import json
    import s3fs
    import zarr
    import ndjson
    from cryoet_data_portal import Client, Run, AnnotationFile
    from copick.impl.filesystem import CopickRootFSSpec
    from copick.models import CopickPoint

    args = get_args()
    dataset_id = args.dataset_id
    copick_config_path = args.copick_config_path
    voxel_spacing_input = float(args.voxel_spacing) if args.voxel_spacing else None
    overlay_root = args.overlay_root or os.path.expanduser("~/copick_overlay")
    static_root = args.static_root or os.path.expanduser("~/copick_static")

    fs = s3fs.S3FileSystem(anon=True)  # Assuming the bucket is public
    data_path = get_data_path()

    # Ensure the data path exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Initialize client
    client = Client()
    
    # Fetch runs
    runs = Run.find(client, [Run.dataset_id == int(dataset_id)])
    if not runs:
        raise ValueError("No runs found for the given dataset ID.")

    # Fetch annotations
    annotations = AnnotationFile.find(client, [AnnotationFile.annotation.tomogram_voxel_spacing.run.dataset_id == int(dataset_id), AnnotationFile.format == 'ndjson'])

    # Generate distinct colors
    max_colors = 32
    distinct_colors = generate_unique_colors(max_colors)

    # Create a new Copick configuration
    pickable_objects = {}
    for annotation in annotations:
        object_name = annotation.annotation.object_name
        if object_name not in pickable_objects:
            label_index = len(pickable_objects)
            color = distinct_colors[label_index % max_colors]
            pickable_objects[object_name] = {
                "name": object_name,
                "is_particle": True,  # Assume all are particles; adjust as needed
                "label": label_index + 1,  # Unique label
                "color": color  # Assign distinct color
            }

    copick_config = {
        "name": "auto_generated_config",
        "description": "Auto-generated config from CryoET data portal",
        "version": "0.1.0",
        "user_id": "albumImport",
        "pickable_objects": list(pickable_objects.values()),
        "overlay_root": overlay_root,
        "static_root": static_root,
        "overlay_fs_args": {},
        "static_fs_args": {}
    }

    # Write the new Copick configuration
    with open(copick_config_path, 'w') as f:
        json.dump(copick_config, f, indent=4)

    # Initialize Copick root with the new configuration
    root = CopickRootFSSpec.from_file(copick_config_path)

    for run in runs:
        run_name = run.name
        print(f"Processing run: {run_name}")

        # Ensure Copick run exists
        copick_run = root.get_run(run_name)
        if not copick_run:
            copick_run = root.new_run(run_name)

        # Ensure voxel spacings and tomograms exist
        for tvs in run.tomogram_voxel_spacings:
            voxel_spacing_value = tvs.voxel_spacing
            if voxel_spacing_input is not None and voxel_spacing_value != voxel_spacing_input:
                continue  # Skip if voxel spacing doesn't match the input
            
            if voxel_spacing_value not in [vs.voxel_size for vs in copick_run.voxel_spacings]:
                voxel_spacing = copick_run.new_voxel_spacing(voxel_size=voxel_spacing_value)
            else:
                voxel_spacing = copick_run.get_voxel_spacing(voxel_spacing_value)

            for tomo in tvs.tomograms:
                s3_zarr_path = tomo.s3_omezarr_dir
                
                tomogram_name = "albumImportFromCryoETDataPortal"
                print(f"Adding tomogram to Copick: {tomogram_name}")

                # Create a new tomogram in Copick
                copick_tomogram = voxel_spacing.new_tomogram(tomogram_name)
                
                # Directly stream data from S3 into the Copick Zarr store
                s3_store = zarr.storage.FSStore(f's3://{s3_zarr_path}', fs=fs)
                copick_store = copick_tomogram.zarr()

                print(f"Streaming data from {s3_zarr_path} to Copick Zarr store for tomogram {tomogram_name}")
                copick_store_group = zarr.group(store=copick_store)
                zarr.copy_all(s3_store, copick_store_group)

        # Process annotations
        for annotation in annotations:
            if annotation.annotation.tomogram_voxel_spacing.run_id != run.id:
                continue
            with fs.open(annotation.s3_path) as pointfile:
                points = list(ndjson.reader(pointfile))
                if not points:
                    continue
                object_name = annotation.annotation.object_name
                label_num = pickable_objects[object_name]["label"]
                user_id = "albumImportFromCryoETDataPortal"
                session_id = "0"
                print(f"Processing annotation for object: {object_name}, label: {label_num}")
                
                centroids = [(p['location']['z'], p['location']['y'], p['location']['x']) for p in points]
                print(f"Saving {len(centroids)} picks for label {label_num}")
                pick_set = copick_run.new_picks(object_name, session_id, user_id)
                pick_set.points = [CopickPoint(location={'x': c[2] * voxel_spacing_value, 'y': c[1] * voxel_spacing_value, 'z': c[0] * voxel_spacing_value}) for c in centroids]
                pick_set.store()

    print(f"Done with all runs. Outputs are saved in the Copick project at {copick_config_path}")

setup(
    group="copick",
    name="project_from_dataportal",
    version="0.1.5",
    title="Fetch Zarr and Annotations from Data Portal and Integrate with Copick",
    description="Fetches Zarr files, annotations, and points from cryoet_data_portal and integrates them into the specified Copick project.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Cellcanvas and Copick teams", "url": "https://cellcanvas.org"}],
    tags=["zarr", "deep learning", "tomography"],
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
