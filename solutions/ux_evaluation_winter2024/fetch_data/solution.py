###album catalog: cellcanvas

from album.runner.api import setup

def run():
    """Run the album solution with partitioned processing."""
    import os
    import pooch
    from album.runner.api import get_args
    import appdirs
    from pathlib import Path
    
    args = get_args()

    # Paths to input files and number of workers
    dataset_name = args.dataset_name

    all_datasets = ["cellcanvas_crop_007.zarr.zip", "cellcanvas_crop_009.zarr.zip", "cellcanvas_crop_010.zarr.zip",]
    
    dataset_names = []
    if dataset_name == "<all>":
        dataset_names = all_datasets
    else:
        dataset_names.append(dataset_name)

    doi = "10.5281/zenodo.10667014"

    download = pooch.DOIDownloader(progressbar=True)       
    
    output_directory = Path(appdirs.user_data_dir("cellcanvas"))
    os.makedirs(output_directory, exist_ok=True)
    print(f"Downloading to {output_directory}")
    
    for dataset_name in dataset_names:        
        if not (output_directory / dataset_name).exists():
            print(f"Downloading {dataset_name}")
            download(
                f"doi:{doi}/{dataset_name}",
                output_file=output_directory / dataset_name,
                pooch=None,
            )
        else:
            print(f"Using cached {dataset_name}")
        

setup(
    group="ux_evaluation_winter2024",
    name="fetch_data",
    version="0.0.3",
    title="Fetch datasets for the UX evaluation",
    description="Fetch datasets for the UX evaluation in Winter 2024.",
    solution_creators=["Kyle Harrington"],
    tags=["data conversion", "DataFrame", "Zarr", "Python"],
    license="MIT",
    covers=[{
        "description": "Cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[
        {"name": "dataset_name", "type": "string", "required": False, "description": "Path for the Zarr file", "default": "<all>"},
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
