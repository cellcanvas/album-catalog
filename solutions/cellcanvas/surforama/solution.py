###album catalog: cellcanvas

from album.runner.api import setup

# from https://github.com/cellcanvas/surforama/blob/main/src/surforama/_cli.py
def run():
    """Run the album solution with partitioned processing."""
    import os
    import pooch
    from album.runner.api import get_args
    import appdirs
    from pathlib import Path
    
    # set up the viewer in demo mode
    from surforama.data._datasets import thylakoid_membrane

    # fetch the data
    tomogram, mesh_data = thylakoid_membrane()

    # Add the data to the viewer
    volume_layer = viewer.add_image(
        tomogram, blending="translucent", depiction="plane"
    )
    surface_layer = viewer.add_surface(mesh_data)

    # set up the slicing plane position
    volume_layer.plane = {"normal": [1, 0, 0], "position": [66, 187, 195]}

    # set up the camera
    viewer.camera.center = (64.0, 124.0, 127.5)
    viewer.camera.zoom = 3.87
    viewer.camera.angles = (
        -5.401480002668876,
        -0.16832643131442776,
        160.28901483338126,
    )    
    
    
setup(
    group="cellcanvas",
    name="surforama",
    version="0.0.1",
    title="Run Surforama",
    description="Run Surforama.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "CellCanvas team.", "url": "https://cellcanvas.org"}],
    tags=["data conversion", "CellCanvas", "Zarr", "Python"],
    license="MIT",
    covers=[{
        "description": "Cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "cellcanvas",
            "name": "cellcanvas",
            "version": "0.0.1",
        }
    },
)
