###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
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
    - textual
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    from pathlib import Path
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Tree
    from textual.widgets.tree import TreeNode
    from copick.impl.filesystem import CopickRootFSSpec
    from rich.text import Text
    from rich.highlighter import ReprHighlighter

    args = get_args()
    copick_config_path = args.copick_config_path

    class CopickTreeApp(App):

        BINDINGS = [
            ("a", "add", "Add node"),
            ("c", "clear", "Clear"),
            ("t", "toggle_root", "Toggle root"),
        ]

        def __init__(self, copick_root):
            super().__init__()
            self.copick_root = copick_root

        def compose(self) -> ComposeResult:
            yield Header()
            yield Footer()
            yield Tree("Copick Project")

        @classmethod
        def add_json(cls, node: TreeNode, json_data: object) -> None:
            """Adds JSON data to a node."""
            highlighter = ReprHighlighter()

            def add_node(name: str, node: TreeNode, data: object) -> None:
                """Adds a node to the tree."""
                if isinstance(data, dict):
                    node.set_label(Text(f"{{}} {name}"))
                    for key, value in data.items():
                        new_node = node.add("")
                        add_node(key, new_node, value)
                elif isinstance(data, list):
                    node.set_label(Text(f"[] {name}"))
                    for index, value in enumerate(data):
                        new_node = node.add("")
                        add_node(str(index), new_node, value)
                else:
                    node.allow_expand = False
                    if name:
                        label = Text.assemble(
                            Text.from_markup(f"[b]{name}[/b]="), highlighter(repr(data))
                        )
                    else:
                        label = Text(repr(data))
                    node.set_label(label)

            add_node("JSON", node, json_data)

        def on_mount(self) -> None:
            """Load and display the Copick project data when the app starts."""
            tree = self.query_one(Tree)
            copick_project_data = self._get_copick_project_data()
            self.add_json(tree.root, copick_project_data)
            tree.root.expand()

        def _get_copick_project_data(self):
            """Get Copick project data as JSON for tree representation."""
            def serialize_copick_object(copick_object):
                return {
                    "name": copick_object.name,
                    "is_particle": copick_object.is_particle,
                    "label": copick_object.label,
                    "color": copick_object.color,
                    "emdb_id": copick_object.emdb_id,
                    "pdb_id": copick_object.pdb_id,
                    "map_threshold": copick_object.map_threshold,
                    "radius": copick_object.radius
                }

            def serialize_copick_run(run):
                return {
                    "meta": run.meta,
                    "voxel_spacings": [serialize_voxel_spacing(vs) for vs in (run.voxel_spacings or [])],
                    "picks": [serialize_pick(p) for p in (run.picks or [])],
                    "meshes": [serialize_mesh(m) for m in (run.meshes or [])],
                    "segmentations": [serialize_segmentation(s) for s in (run.segmentations or [])]
                }

            def serialize_voxel_spacing(voxel_spacing):
                return {
                    "meta": voxel_spacing.meta,
                    "tomograms": [serialize_tomogram(t) for t in (voxel_spacing.tomograms or [])]
                }

            def serialize_tomogram(tomogram):
                return {
                    "meta": tomogram.meta,
                    "features": [serialize_feature(f) for f in (tomogram.features or [])],
                    "tomo_type": tomogram.tomo_type
                }

            def serialize_feature(feature):
                return {
                    "meta": feature.meta,
                    "tomo_type": feature.tomo_type,
                    "feature_type": feature.feature_type
                }

            def serialize_pick(pick):
                return {
                    "meta": pick.meta,
                    "points": pick.points,
                    "from_tool": pick.from_tool,
                    "pickable_object_name": pick.pickable_object_name,
                    "user_id": pick.user_id,
                    "session_id": pick.session_id,
                    "trust_orientation": pick.trust_orientation,
                    "color": pick.color
                }

            def serialize_mesh(mesh):
                return {
                    "meta": mesh.meta,
                    "mesh": mesh.mesh,
                    "from_tool": mesh.from_tool,
                    "from_user": mesh.from_user,
                    "pickable_object_name": mesh.pickable_object_name,
                    "user_id": mesh.user_id,
                    "session_id": mesh.session_id,
                    "color": mesh.color
                }

            def serialize_segmentation(segmentation):
                # Check for None and provide default values
                color = None
                try:
                    color = segmentation.color
                except AttributeError:
                    color = [128, 128, 128, 0] if segmentation.is_multilabel else None

                return {
                    "meta": segmentation.meta,
                    "zarr": segmentation.zarr,
                    "from_tool": segmentation.from_tool,
                    "from_user": segmentation.from_user,
                    "user_id": segmentation.user_id,
                    "session_id": segmentation.session_id,
                    "is_multilabel": segmentation.is_multilabel,
                    "voxel_size": segmentation.voxel_size,
                    "name": segmentation.name,
                    "color": color
                }

            data = {
                "config": self.copick_root.config,
                "user_id": self.copick_root.user_id,
                "session_id": self.copick_root.session_id,
                "runs": [serialize_copick_run(run) for run in (self.copick_root.runs or [])],
                "pickable_objects": [serialize_copick_object(obj) for obj in (self.copick_root.pickable_objects or [])]
            }
            return data

        def action_add(self) -> None:
            """Add a node to the tree."""
            tree = self.query_one(Tree)
            copick_project_data = self._get_copick_project_data()
            json_node = tree.root.add("Copick Project")
            self.add_json(json_node, copick_project_data)
            tree.root.expand()

        def action_clear(self) -> None:
            """Clear the tree (remove all nodes)."""
            tree = self.query_one(Tree)
            tree.clear()

        def action_toggle_root(self) -> None:
            """Toggle the root node."""
            tree = self.query_one(Tree)
            tree.show_root = not tree.show_root

    copick_root = CopickRootFSSpec.from_file(copick_config_path)

    CopickTreeApp(copick_root).run()


setup(
    group="copick",
    name="display-copick-index",
    version="0.0.2",
    title="Display Copick Project Index",
    description="A solution that opens a Copick project and displays the index using textual.",
    solution_creators=["Kyle Harrington"],
    tags=["copick", "textual", "index"],
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
