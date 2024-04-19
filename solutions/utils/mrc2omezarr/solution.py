###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - mrcfile
  - h5py
  - numpy
  - pip
  - zarr
  - pip:
    - git+https://github.com/uermel/mrc2omezarr.git
"""

def run():
    from mrc2omezarr.proc import convert_mrc_to_ngff
    from mrc2omezarr.util import get_filesystem_args
    
    args = get_args()
    
    scale_factors = tuple([int(x) for x in args.scale_factors.split(",")]) if args.scale_factors else None
    voxel_size = [float(x) for x in args.voxel_size.split(",")] if args.voxel_size else None
    filesystem_args = get_filesystem_args(args.filesystem_args) if args.filesystem_args else None
    
    convert_mrc_to_ngff(
        args.mrc_path,
        args.zarr_path,
        args.permissive,
        args.overwrite,
        scale_factors,
        voxel_size,
        args.is_image_stack,
        (args.chunk_size, args.chunk_size, args.chunk_size) if args.chunk_size else None,
        filesystem_args,
        args.pyramid_method,
    )
    
    print(f"Conversion complete: {args.mrc_path} to {args.zarr_path}")

setup(
    group="utils",
    name="mrc2omezarr",
    version="0.0.3",
    title="Convert a mrc to omezarr using mrc2omezarr",
    description="Convert a mrc to omezarr using mrc2omezarr.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Utz Ermel.", "url": "https://github.com/uermel/mrc2omezarr"}],
    tags=["mrc", "ome-zarr", "zarr"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "mrc_path", "type": "string", "required": True, "description": "Path to the MRC file. Include the protocol if necessary (e.g., s3://)."},
        {"name": "zarr_path", "type": "string", "required": True, "description": "Path to the output Zarr file. Include the protocol if necessary (e.g., s3://)."},
        {"name": "permissive", "type": "boolean", "required": False, "description": "Whether to read the MRC file in permissive mode.", "default": False},
        {"name": "overwrite", "type": "boolean", "required": False, "description": "Whether to overwrite the output Zarr file.", "default": False},
        {"name": "scale_factors", "type": "string", "required": False, "description": "Scale factors for multiscale pyramid. Comma-separated list of integers.", "default": "1,2,4"},
        {"name": "voxel_size", "type": "string", "required": False, "description": "Voxel size in Angstroms. Comma-separated list of floats or a single float. If not provided, it will be read from the MRC header."},
        {"name": "is_image_stack", "type": "boolean", "required": False, "description": "Whether the data is an image stack (determined from MRC-header by default)."},
        {"name": "chunk_size", "type": "integer", "required": False, "description": "Chunk size for the Zarr file.", "default": 256},
        {"name": "filesystem_args", "type": "string", "required": False, "description": "Path to a JSON file containing additional arguments to pass to the fsspec-filesystem."},
        {"name": "pyramid_method", "type": "string", "required": False, "description": "Method to downscale the data. Options: local_mean, downsample.", "default": "local_mean"},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
