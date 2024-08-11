###album catalog: cellcanvas

from io import StringIO
import subprocess
from album.runner.api import setup

env_file = StringIO(
    """name: copick_live
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - pybind11
  - pip
  - numpy
  - pandas
  - pillow
  - redis==3.5.3
  - redis-server
  - celery==5.2.3
  - kombu==5.2.3
  - sqlalchemy
  - psycopg2
  - album    
  - flask==2.2.5
  - pip:
      - dash==2.13.0
      - plotly==5.17.0
      - dash-extensions==1.0.1
      - dash-bootstrap-components==1.5.0
      - dash-iconify==0.1.2
      - apscheduler
      - paramiko
      - git+https://github.com/uermel/copick.git
      - album
      - "git+https://github.com/kephale/copick_live.git@refactor"
      - python-multipart
"""
)


def run():
    from album.runner.api import get_args
    import time
    from copick_live.config import get_config

    args = get_args()
    config_path = args.config_path

    # Update global state with dataset configurations
    config = get_config(config_path)

    if config.album_mode:
        # Start redis-server
        redis_server_process = subprocess.Popen(['redis-server'])
        time.sleep(5)  # Give Redis server some time to start

        # Start Celery worker
        celery_worker_process = subprocess.Popen(
            ['celery', '-A', 'copick_live.celery_tasks', 'worker', '--loglevel=info']
        )
        time.sleep(5)  # Give Celery worker some time to start

    # Import and run Dash app
    from copick_live.app import create_app
    dash_app = create_app()
    dash_app.run_server(host="0.0.0.0", port=8000, debug=False)

    if config.album_mode:
        # Ensure all processes are terminated correctly when Dash app stops
        redis_server_process.terminate()
        celery_worker_process.terminate()


setup(
    group="copick",
    name="copick_live",
    version="0.0.4",
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
