###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - napari
  - pyqt
  - numpy
  - pip:
    - album
    - "git+https://github.com/kephale/napari-copick.git"
"""

def run():
    import os
    from napari import Viewer
    from napari_copick._widget import CopickPlugin
    import napari

    # Initialize the Napari viewer
    viewer = Viewer()

    # Add Copick plugin widget to the viewer
    viewer.window.add_dock_widget(CopickPlugin(viewer))

    # Start Napari
    napari.run()

setup(
    group="copick",
    name="napari-copick",
    version="0.0.1",
    title="Napari Copick Plugin Launcher",
    description="A solution that installs napari-copick and launches the CopickPlugin.",
    solution_creators=["Kyle Harrington"],
    tags=["napari", "copick", "plugin", "visualization"],
    license="MIT",
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
