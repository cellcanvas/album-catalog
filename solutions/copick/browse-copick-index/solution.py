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
  - joblib
  - paramiko
  - scikit-learn==1.3.2
  - pip:
    - album
    - panel
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import subprocess
    import os
    import sys
    import json
    import panel as pn
    import param
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()
    copick_config_path = args.copick_config_path

    # Initialize Panel
    pn.extension()

    # Load Copick project
    copick_root = CopickRootFSSpec.from_file(copick_config_path)

    class CopickApp(param.Parameterized):
        tomogram_path = param.String(default='')

        def __init__(self, **params):
            super().__init__(**params)
            self.intro = pn.pane.Markdown("# Copick Project Index", width=800)
            self.links = pn.pane.Markdown(self.get_tomogram_links(), width=800)
            self.tomogram_path_input = pn.widgets.TextInput(name="Enter the Zarr path to open a tomogram:")
            self.open_button = pn.widgets.Button(name="Open Tomogram", button_type="primary")
            self.status = pn.pane.Markdown("", width=800)
            self.open_button.on_click(self.open_tomogram)

        def extract_zarr_path(self, zarr_store):
            """Extracts the path from the Zarr store."""
            return str(zarr_store.path)
        
        def get_tomogram_links(self):
            """Generate hierarchical structure for the Panel app."""
            links = []
            for run in copick_root.runs:
                run_info = {
                    "meta": run.meta,
                    "voxel_spacings": [vs.meta for vs in run.voxel_spacings],
                    "picks": [pick.meta for pick in run.picks],
                    "meshes": [mesh.meta for mesh in run.meshes],
                    "segmentations": [seg.meta for seg in run.segmentations]
                }
                run_html = f"<b>Run:</b> {run.meta}"
                run_links = []
                for voxel_spacing in run.voxel_spacings:
                    voxel_data = {
                        "meta": voxel_spacing.meta,
                        "tomograms": [{"meta": tomo.meta, "zarr": self.extract_zarr_path(tomo.zarr())} for tomo in voxel_spacing.tomograms]
                    }
                    voxel_html = f"<b>Voxel Spacing:</b> {voxel_spacing.meta}"
                    tomogram_links = []
                    for tomogram in voxel_spacing.tomograms:
                        zarr_path = self.extract_zarr_path(tomogram.zarr())
                        tomogram_link = f'<a href="/open_tomogram?path={zarr_path}" target="_blank">Tomogram: {zarr_path}</a>'
                        tomogram_links.append(tomogram_link)
                    run_links.append((voxel_html, tomogram_links))
                links.append((run_html, run_links))
            markdown_content = ""
            for run_html, run_links in links:
                markdown_content += f"{run_html}\n"
                for voxel_html, tomogram_links in run_links:
                    markdown_content += f"{voxel_html}\n"
                    for tomogram_link in tomogram_links:
                        markdown_content += f"{tomogram_link}\n"
            return markdown_content

        def open_tomogram(self, event=None):
            tomogram_path = self.tomogram_path_input.value
            if tomogram_path:
                try:
                    json_args = json.dumps({"zarr": tomogram_path})
                    subprocess.run(["album", "run", "copick:napari-copick:0.0.1", "--args", json_args], check=True)
                    self.status.object = f"Opened tomogram at {tomogram_path}"
                except subprocess.CalledProcessError as e:
                    self.status.object = f"Failed to open tomogram: {str(e)}"
            else:
                self.status.object = "Please enter a valid Zarr path."

        @pn.depends('tomogram_path')
        def view(self):
            return pn.Column(
                self.intro,
                self.links,
                self.tomogram_path_input,
                self.open_button,
                self.status
            )

    app = CopickApp()

    # Add route for open_tomogram
    @pn.depends(path=pn.state.location.param.query)
    def open_tomogram_from_url(path):
        if path:
            app.tomogram_path = path
            app.open_tomogram()
            return pn.Column(pn.pane.Markdown("# Tomogram opened from URL"))
        return pn.Column()

    # Serve the Panel app
    pn.serve({'/': app.view, '/open_tomogram': open_tomogram_from_url}, start=True, show=True)

setup(
    group="copick",
    name="browse-copick-index",
    version="0.0.2",
    title="Display Copick Project Index",
    description="A solution that opens a Copick project and displays the index using Panel.",
    solution_creators=["Kyle Harrington"],
    tags=["copick", "panel", "index"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
