###album catalog: cellcanvas

from io import StringIO
from pathlib import Path

from album.runner.api import setup

env_file = StringIO(
    """name: copick_live
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pybind11
  - pip
  - numpy
  - pandas
  - pillow
  - flask==2.2.5
  - pip:
      - dash==2.13.0
      - plotly==5.17.0
      - dash-extensions==1.0.1
      - dash-bootstrap-components==1.5.0
      - dash-iconify==0.1.2
      - apscheduler
      - git+https://github.com/uermel/copick.git
      - git+https://github.com/kephale/copick_live.git@refactor
"""
)


def run():
    import os
    import configparser
    import dash_bootstrap_components as dbc
    from dash import Dash, html, dcc
    from collections import defaultdict
    from album.runner.api import get_args

    # EEK this is order sensitive
    from copick_live.utils.copick_dataset import get_copick_dataset
    from copick_live.utils.local_dataset import get_local_dataset

    args = get_args()
    config_path = args.config_path
    
    # hack to load with correct config
    get_copick_dataset(config_path)
    get_local_dataset(config_path)
    
    from copick_live.components.header import layout as header
    from copick_live.components.progress import layout as tomo_progress
    from copick_live.components.proteins import layout as protein_sts
    from copick_live.components.waitlist import layout as unlabelled_tomos
    from copick_live.components.annotators import layout as ranking
    from copick_live.components.composition import layout as composition
    from copick_live.components.popups import layout as popups

    def create_app():
        external_stylesheets = [dbc.themes.BOOTSTRAP,
                                "/assets/header-style.css",
                                "https://codepen.io/chriddyp/pen/bWLwgP.css",
                                "https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        css_content = """
        #app-title > * {
            color: white;
            margin: 0;
            padding: 0;
        }

        #howto-open {
            white-space: nowrap;
        }

        #gh-link {
            white-space: nowrap;
        }

        .card {
            margin: 1em 0 1em 0;
            min-width: auto;
        }

        label {
          display: block;
          margin-bottom: 0;
        }

        #transparent-loader-wrapper > div {
            visibility: visible !important;
        }
        """

        assets_path = Path("assets")
        assets_path.mkdir(exist_ok=True)
        css_file = assets_path / "header-style.css"
        css_file.write_text(css_content)

        browser_cache = html.Div(
            id="no-display",
            children=[
                dcc.Interval(
                    id='interval-component',
                    interval=20*1000,  # clientside check in milliseconds, 10s
                    n_intervals=0
                ),
                dcc.Store(id='tomogram-index', data=''),
                dcc.Store(id='keybind-num', data=''),
                dcc.Store(id='run-dt', data=defaultdict(list))
            ],
        )

        app.layout = html.Div(
            [
                header(),
                popups(),
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col([tomo_progress(),
                                         unlabelled_tomos()
                                         ],
                                        width=3),
                                dbc.Col(ranking(), width=3),
                                dbc.Col(composition(), width=3),
                                dbc.Col(protein_sts(), width=3),
                            ],
                            justify='center',
                            className="h-100",
                        ),
                    ],
                    fluid=True,
                ),
                html.Div(browser_cache)
            ],
        )
        return app

    dash_app = create_app()
    dash_app.run_server(host="0.0.0.0", port=8000, debug=False)

setup(
    group="copick",
    name="copick_live",
    version="0.0.2",
    title="Run CoPick Live.",
    description="Run CoPick Live",
    solution_creators=["Zhuowen Zhao and Kyle Harrington"],
    cite=[{"text": "CoPick Live team.", "url": "https://github.com/zhuowenzhao/copick_live"}],
    tags=["imaging", "cryoet", "Python", "napari", "copick_live"],
    license="MIT",
    covers=[
        {
            "description": "CoPick Live screenshot.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    args=[
        {
            "name": "config_path",
            "type": "string",
            "description": "Path to the configuration file.",
            "required": False
        }
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
