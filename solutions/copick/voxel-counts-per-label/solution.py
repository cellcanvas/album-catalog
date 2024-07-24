###album catalog: cellcanvas

from album.runner.api import setup, get_args

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pandas
  - zarr
  - numpy
  - scipy
  - scikit-image
  - dask
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    import numpy as np
    import pandas as pd
    import zarr
    from copick.impl.filesystem import CopickRootFSSpec
    import json
    from jinja2 import Template

    args = get_args()
    copick_config_path = args.copick_config_path
    session_id = args.session_id
    user_id = args.user_id
    segmentation_name = args.segmentation_name
    output_csv = args.output_csv
    output_html = args.output_html

    root = CopickRootFSSpec.from_file(copick_config_path)
    
    runs = root.runs
    pickable_objects = {obj.label: obj.name for obj in root.pickable_objects}
    
    data = {name: [] for name in pickable_objects.values()}

    def load_segmentation(run, segmentation_name):
        segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=segmentation_name)
        if not segs:
            raise FileNotFoundError(f"No segmentation found with name: {segmentation_name}")
        seg = segs[0]
        return zarr.open(seg.path, mode='r')['data'][:]
    
    for run in runs:
        try:
            segmentation = load_segmentation(run, segmentation_name)
            unique_labels, counts = np.unique(segmentation, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))

            for label, name in pickable_objects.items():
                count = label_counts.get(label, 0)
                data[name].append(count)
        except FileNotFoundError as e:
            print(f"Skipping run {run.name} due to missing segmentation: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voxel Counts</title>
        <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.24/css/jquery.dataTables.css">
    </head>
    <body>
        <h1>Voxel Counts</h1>
        <table id="voxelCounts" class="display">
            <thead>
                <tr>
                    {% for column in columns %}
                    <th>{{ column }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                <tr>
                    {% for value in row %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <script>
            $(document).ready( function () {
                $('#voxelCounts').DataTable();
            });
        </script>
    </body>
    </html>
    """)

    html_content = template.render(columns=df.columns, rows=df.values)
    with open(output_html, 'w') as f:
        f.write(html_content)

    print(f"CSV saved to {output_csv}")
    print(f"HTML saved to {output_html}")

setup(
    group="copick",
    name="voxel-counts-per-label",
    version="0.1.0",
    title="Voxel Counts per Label",
    description="A solution that counts the number of voxels per label in a segmentation and saves the results as a CSV and HTML page.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "zarr", "segmentation", "voxels", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID for the segmentation."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID for segmentation creation."},
        {"name": "segmentation_name", "type": "string", "required": True, "description": "Name of the segmentation to process."},
        {"name": "output_csv", "type": "string", "required": True, "description": "Output path for the CSV file."},
        {"name": "output_html", "type": "string", "required": True, "description": "Output path for the HTML file."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
