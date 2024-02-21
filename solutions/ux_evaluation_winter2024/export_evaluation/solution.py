###album catalog: cellcanvas

from album.runner.api import setup

def run():
    from album.runner.api import get_args
    import appdirs
    from pathlib import Path
    import zipfile
    import logging
    import sys
    import os

    dataset_name = get_args().dataset_name
    
    all_datasets = ["cellcanvas_crop_007.zarr.zip", "cellcanvas_crop_008.zarr.zip", "cellcanvas_crop_009.zarr.zip", "cellcanvas_crop_010.zarr.zip", "cellcanvas_crop_011.zarr.zip"]

    if dataset_name not in all_datasets:
        print(f"Dataset {dataset_name} invalid. Select from {all_datasets}")
        return
        
    data_directory = Path(appdirs.user_data_dir("cellcanvas"))

    zip_path = data_directory / f"{dataset_name}.zarr.zip"

    zarr_directory = data_directory / f"{dataset_name}.zarr"

    # Unzip the Zarr archive if the directory doesn't exist
    if not zarr_directory.exists():
        print(f"zarr {zarr_directory} doesn't exist. Failing.")
        return

    # Set up logging to file in the same directory as the zarr
    log_file_path = zarr_directory / "cellcanvas.log"

    home_directory = Path.home()
    documents_directory = home_directory / "Documents"
    documents_directory.mkdir(parents=True, exist_ok=True)
    export_path = documents_directory / f"{dataset_name}_export.zip"
            
    with zipfile.ZipFile(export_path, 'w') as zipf:

        for subdir in ["prediction", "painting"]:
            subdir_path = zarr_directory / subdir
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, arcname=file_path.relative_to(zarr_directory))
        
        # Add cellcanvas.log to the zip
        zipf.write(log_file_path, arcname=log_file_path.relative_to(zarr_directory))

    print(f"Exported results to {export_path}")


setup(
    group="ux_evaluation_winter2024",
    name="export_evaluation",
    version="0.0.4",
    title="Export results of the UX evaluation",
    description="Export results of the UX evaluation.",
    solution_creators=["Kyle Harrington"],
    tags=["data conversion", "DataFrame", "Zarr", "Python"],
    license="MIT",
    covers=[{
        "description": "Cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[
        {"name": "dataset_name", "type": "string", "required": True, "description": "Path for the Zarr file"},
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "ux_evaluation_winter2024",
            "name": "python-nographics",
            "version": "0.0.4",
        }
    },
)
