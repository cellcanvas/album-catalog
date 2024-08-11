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
    - textual
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import sys
    import termios
    import atexit
    from pathlib import Path
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Tree
    from textual.widgets.tree import TreeNode
    from copick.impl.filesystem import CopickRootFSSpec
    from rich.text import Text
    from rich.highlighter import ReprHighlighter
    import logging

    # Configure logging to only show errors
    logging.basicConfig(level=logging.ERROR)

    album_logger = logging.getLogger("album")
    album_logger.setLevel(logging.ERROR)
    
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
            self.node_data = {}  # Dictionary to store node data

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

        def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
            """Handle the tree node expanded event to load data lazily."""
            node = event.node
            node_id = node.tree_node_id
            if node_id in self.node_data:
                data_type, data = self.node_data[node_id]
                if data_type == "run":
                    self.add_run_data_nodes(node, data)
                elif data_type == "voxel":
                    self.add_voxel_spacing_data_nodes(node, data)
                elif data_type == "tomogram":
                    self.add_tomogram_data_nodes(node, data)
                elif data_type == "segmentation":
                    self.add_segmentation_data_nodes(node, data)

        def add_runs_node(self, root_node: TreeNode) -> None:
            """Add runs node to the root node."""
            for i, run in enumerate(self.copick_root.runs):
                run_node = root_node.add(f"Run: {run.meta}")
                run_node_id = f"run_{i}"
                run_node.tree_node_id = run_node_id
                self.node_data[run_node_id] = ("run", run)

        def add_run_data_nodes(self, run_node: TreeNode, run) -> None:
            """Add voxel spacings, picks, meshes, and segmentations nodes to the run node."""
            voxel_node = run_node.add("Voxel Spacings")
            voxel_node_id = f"voxel_{run.meta}"
            voxel_node.tree_node_id = voxel_node_id
            self.node_data[voxel_node_id] = ("voxel", run.voxel_spacings)

            picks_node = run_node.add("Picks")
            picks_node_id = f"picks_{run.meta}"
            picks_node.tree_node_id = picks_node_id
            self.node_data[picks_node_id] = ("picks", run.picks)

            meshes_node = run_node.add("Meshes")
            meshes_node_id = f"meshes_{run.meta}"
            meshes_node.tree_node_id = meshes_node_id
            self.node_data[meshes_node_id] = ("meshes", run.meshes)

            segmentations_node = run_node.add("Segmentations")
            segmentations_node_id = f"segmentation_{run.meta}"
            segmentations_node.tree_node_id = segmentations_node_id
            self.node_data[segmentations_node_id] = ("segmentation", run.segmentations)

        def add_voxel_spacing_data_nodes(self, voxel_node: TreeNode, voxel_spacings) -> None:
            """Add tomograms node to the voxel spacing node."""
            for i, voxel_spacing in enumerate(voxel_spacings):
                tomogram_node = voxel_node.add(f"Tomogram: {voxel_spacing.meta}")
                tomogram_node_id = f"tomogram_{i}"
                tomogram_node.tree_node_id = tomogram_node_id
                self.node_data[tomogram_node_id] = ("tomogram", voxel_spacing.tomograms)

        def add_tomogram_data_nodes(self, tomogram_node: TreeNode, tomograms) -> None:
            """Add features node to the tomogram node."""
            for i, tomogram in enumerate(tomograms):
                for feature in tomogram.features:
                    feature_node = tomogram_node.add(f"Feature: {feature.meta}")
                    feature_node_id = f"feature_{i}"
                    feature_node.tree_node_id = feature_node_id
                    self.node_data[feature_node_id] = ("feature", feature)

        def add_segmentation_data_nodes(self, segmentation_node: TreeNode, segmentations) -> None:
            """Add segmentation details to the segmentation node."""
            for i, segmentation in enumerate(segmentations):
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

    def reset_terminal():
        if os.name == 'posix':
            sys.stdout.write("\x1b[?1003l\x1b[?1006l\x1b[?1015l")
            sys.stdout.flush()
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))

    atexit.register(reset_terminal)

    logging.basicConfig(level=logging.ERROR)

    # Silence all other loggers except for errors
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    CopickTreeApp(copick_root).run()

setup(
    group="copick",
    name="display-copick-index",
    version="0.0.9",
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
