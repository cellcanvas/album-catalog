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
            """Initialize the tree with the root node."""
            tree = self.query_one(Tree)
            tree.root.tree_node_id = "root"
            self.add_runs_node(tree.root)

        def on_tree_node_expanded(self, event: Tree.TreeNodeExpanded) -> None:
            """Handle the tree node expanded event to load data lazily."""
            node = event.node
            if not node.children:
                if node.tree_node_id.startswith("run"):
                    self.add_run_data_nodes(node)
                elif node.tree_node_id.startswith("voxel"):
                    self.add_voxel_spacing_data_nodes(node)
                elif node.tree_node_id.startswith("tomogram"):
                    self.add_tomogram_data_nodes(node)
                elif node.tree_node_id.startswith("segmentation"):
                    self.add_segmentation_data_nodes(node)

        def add_runs_node(self, root_node: TreeNode) -> None:
            """Add runs node to the root node."""
            for run in self.copick_root.runs:
                run_node = root_node.add(f"Run: {run.meta}")
                run_node.tree_node_id = f"run_{run.meta}"

        def add_run_data_nodes(self, run_node: TreeNode) -> None:
            """Add voxel spacings, picks, meshes, and segmentations nodes to the run node."""
            run = self.copick_root.get_run(run_node.tree_node_id.split("_")[1])
            if run:
                voxel_node = run_node.add("Voxel Spacings")
                voxel_node.tree_node_id = f"voxel_{run.meta}"

                picks_node = run_node.add("Picks")
                picks_node.tree_node_id = f"picks_{run.meta}"

                meshes_node = run_node.add("Meshes")
                meshes_node.tree_node_id = f"meshes_{run.meta}"

                segmentations_node = run_node.add("Segmentations")
                segmentations_node.tree_node_id = f"segmentation_{run.meta}"

        def add_voxel_spacing_data_nodes(self, voxel_node: TreeNode) -> None:
            """Add tomograms node to the voxel spacing node."""
            run_meta = voxel_node.tree_node_id.split("_")[1]
            run = self.copick_root.get_run(run_meta)
            if run:
                for voxel_spacing in run.voxel_spacings:
                    tomogram_node = voxel_node.add(f"Tomogram: {voxel_spacing.meta}")
                    tomogram_node.tree_node_id = f"tomogram_{voxel_spacing.meta}"

        def add_tomogram_data_nodes(self, tomogram_node: TreeNode) -> None:
            """Add features node to the tomogram node."""
            run_meta = tomogram_node.tree_node_id.split("_")[1]
            run = self.copick_root.get_run(run_meta)
            if run:
                for voxel_spacing in run.voxel_spacings:
                    for tomogram in voxel_spacing.tomograms:
                        if tomogram.meta == tomogram_node.tree_node_id.split("_")[1]:
                            for feature in tomogram.features:
                                feature_node = tomogram_node.add(f"Feature: {feature.meta}")
                                feature_node.tree_node_id = f"feature_{feature.meta}"

        def add_segmentation_data_nodes(self, segmentation_node: TreeNode) -> None:
            """Add segmentation details to the segmentation node."""
            run_meta = segmentation_node.tree_node_id.split("_")[1]
            run = self.copick_root.get_run(run_meta)
            if run:
                for segmentation in run.segmentations:
                    if segmentation.meta == segmentation_node.tree_node_id.split("_")[1]:
                        try:
                            color = segmentation.color
                        except AttributeError:
                            color = [128, 128, 128, 0] if segmentation.is_multilabel else None

                        segmentation_data = {
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
                        self.add_json(segmentation_node, segmentation_data)

    copick_root = CopickRootFSSpec.from_file(copick_config_path)

    CopickTreeApp(copick_root).run()


setup(
    group="copick",
    name="display-copick-index",
    version="0.0.3",
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
