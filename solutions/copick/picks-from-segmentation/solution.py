###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - zarr
  - numpy
  - scipy
  - scikit-image
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import numpy as np
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec
    import scipy.ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.measure import regionprops
    from copick.models import CopickPoint
    from skimage.morphology import binary_erosion, binary_dilation, ball
    import json

    args = get_args()
    copick_config_path = args.copick_config_path
    painting_segmentation_name = args.painting_segmentation_name
    session_id = args.session_id
    user_id = args.user_id
    voxel_spacing = int(args.voxel_spacing)
    maxima_filter_size = int(args.maxima_filter_size)
    run_name = args.run_name
    segmentation_dir = args.segmentation_dir
    min_particle_size = int(args.min_particle_size)
    max_particle_size = int(args.max_particle_size)
    segmentation_idx_offset = int(args.segmentation_idx_offset)

    labels_to_process = list(map(int, args.labels_to_process.split(',')))
    
    root = CopickRootFSSpec.from_file(copick_config_path)
    run = root.get_run(run_name)

    def get_painting_segmentation(run, user_id, session_id, painting_segmentation_name, voxel_spacing):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
        if not run.get_voxel_spacing(voxel_spacing).get_tomogram("denoised"):
            return None
        elif len(segs) == 0:
            seg = run.new_segmentation(
                voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
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

    def load_multilabel_segmentation(segmentation_dir, segmentation_name):
        print(f"Loading segmentation from {segmentation_dir} with name {segmentation_name}")
        segmentation_file = [f for f in os.listdir(segmentation_dir) if f.endswith('.zarr') and segmentation_name in f]
        if not segmentation_file:
            raise FileNotFoundError(f"No segmentation file found with name: {segmentation_name}")
        seg_path = os.path.join(segmentation_dir, segmentation_file[0])
        print(f"Loaded segmentation from {seg_path}")
        return (zarr.open(seg_path, mode='r')['data'][:] + segmentation_idx_offset)

    def detect_local_maxima(distance):
        footprint = np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))
        local_max = (distance == ndi.maximum_filter(distance, footprint=footprint))
        return local_max

    def get_centroids_and_save(segmentation, labels, run, user_id, session_id, voxel_spacing):
        all_centroids = {}
        edt_results = {}
        watershed_results = {}

        # Structuring element for erosion and dilation
        struct_elem = ball(1)  # Adjust the size of the ball as needed

        # Create a binary mask where particles are detected based on labels_to_process
        print("Creating binary mask for particles...")
        binary_mask = np.isin(segmentation, labels_to_process).astype(int)
        eroded = binary_erosion(binary_mask, struct_elem)
        dilated = binary_dilation(eroded, struct_elem)
        print("Binary mask created.")

        print("Calculating distance transform...")
        distance = ndi.distance_transform_edt(dilated)
        edt_results['combined'] = distance
        print("Distance transform calculated.")

        print("Detecting local maxima...")
        local_maxi = detect_local_maxima(distance)
        print("Local maxima detected.")

        print("Labeling markers...")
        markers, _ = ndi.label(local_maxi)
        print(f"Markers labeled: {np.unique(markers)}")

        print("Applying watershed segmentation...")
        watershed_labels = watershed(-distance, markers, mask=dilated)
        watershed_results['combined'] = watershed_labels
        print("Watershed segmentation applied.")

        print(f"Starting to process {len(np.unique(watershed_labels))} individual watershed regions for labeling")

        unique_watershed_labels = np.unique(watershed_labels)
        props_list = regionprops(watershed_labels)

        print("Computed region statistics")
        
        # Precompute the unique segmentation labels and their counts once
        unique_segmentation_labels = np.unique(segmentation)
        segmentation_counts = np.bincount(segmentation.flat)

        print("Computed segmentation counts. Starting per region processing...")

        # Iterate through unique labels in watershed
        for region in props_list:
            label_num = region.label
            if label_num == 0:
                continue  # Skip background
            
            # Mask for the current watershed region
            region_mask = watershed_labels == label_num
            
            # Analyze the distribution of original segmentation labels within this watershed region
            original_labels_in_region = segmentation[region_mask]

            if len(original_labels_in_region) == 0:
                continue  # Skip empty regions

            # Determine the most frequent label
            unique_labels, counts = np.unique(original_labels_in_region, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            
            # Use centroid of the region to assign a pick
            centroid = region.centroid
            if min_particle_size <= region.area <= max_particle_size:
                if dominant_label in all_centroids:
                    all_centroids[dominant_label].append(centroid)
                else:
                    all_centroids[dominant_label] = [centroid]

        return all_centroids, edt_results, watershed_results
    
    def save_centroids_as_picks(run, user_id, session_id, voxel_spacing, all_centroids):
        print("Saving centroids for all labels...")
        for label_num, centroids in all_centroids.items():
            object_name = [obj.name for obj in root.pickable_objects if obj.label == label_num]
            if not object_name:
                print(f"No object name found for label {label_num}.")
                raise ValueError(f"Label {label_num} does not correspond to any object name in pickable objects.")
            object_name = object_name[0]
            pick_set = run.new_picks(object_name, session_id, user_id)
            pick_set.points = [CopickPoint(location={'x': c[2] * voxel_spacing, 'y': c[1] * voxel_spacing, 'z': c[0] * voxel_spacing}) for c in centroids]
            pick_set.store()
            print(f"Saved {len(centroids)} centroids for label {label_num} {object_name}.")

    print(f"Processing run {run_name}")

    multilabel_segmentation = load_multilabel_segmentation(segmentation_dir, painting_segmentation_name)

    print("Starting to find centroids")
    centroids, edt_results, watershed_results = get_centroids_and_save(multilabel_segmentation, labels_to_process, run, user_id, session_id, voxel_spacing)

    save_centroids_as_picks(run, user_id, session_id, voxel_spacing, centroids)

    print("Centroid extraction and saving complete.")

setup(
    group="copick",
    name="picks-from-segmentation",
    version="0.0.22",
    title="Extract Centroids from Multilabel Segmentation",
    description="A solution that extracts centroids from a multilabel segmentation using Copick and saves them as candidate picks.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "segmentation", "centroids", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "painting_segmentation_name", "type": "string", "required": True, "description": "Name of the painting segmentation."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "voxel_spacing", "type": "integer", "required": True, "description": "Voxel spacing used to scale pick locations."},
        {"name": "run_name", "type": "string", "required": True, "description": "Name of the Copick run to process."},
        {"name": "segmentation_dir", "type": "string", "required": True, "description": "Directory containing the multilabel segmentation."},
        {"name": "min_particle_size", "type": "integer", "required": False, "description": "Minimum size threshold for particles.", "default": 1000},
        {"name": "max_particle_size", "type": "integer", "required": False, "description": "Maximum size threshold for particles.", "default": 50000},
        {"name": "maxima_filter_size", "type": "integer", "required": False, "description": "Size for the maximum detection filter (default 9).", "default": 9},
        {"name": "segmentation_idx_offset", "type": "integer", "required": False, "description": "Offset for segmentation indices (default 0).", "default": 0},
        {"name": "labels_to_process", "type": "string", "required": True, "description": "Comma-separated list of labels to process."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
