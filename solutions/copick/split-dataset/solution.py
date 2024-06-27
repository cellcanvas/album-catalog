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
    import zarr
    import json
    from copick.impl.filesystem import CopickRootFSSpec
    from collections import defaultdict
    import copy
    import random
    from scipy.stats import ks_2samp

    args = get_args()
    copick_config_path = args.copick_config_path
    ks_values = list(map(float, args.ks_values.split(',')))
    output_json_path = args.output_json_path
    specific_user_id = args.user_id
    specific_session_id = args.session_id

    root = CopickRootFSSpec.from_file(copick_config_path)

    class SplitDataset:
        def __init__(self, root, specific_user_id, specific_session_id):
            self.root = root
            self.arrs = []
            self.tomograms = []

            N = len(self.root.runs)
            self.particle_map = {o.name: i for i, o in enumerate(self.root.config.pickable_objects)}
            self.run_stats_list = [0] * N
            for i, run in enumerate(self.root.runs[:N]):
                self.tomograms.append(run.name)
                counter = defaultdict(int)
                for pick in run.picks:
                    if pick.user_id == specific_user_id and pick.session_id == specific_session_id:
                        try:
                            if pick.points is not None:
                                counter[pick.pickable_object_name] = len(pick.points)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Error reading pick points for {pick.pickable_object_name} in run {run.name}: {e}")
                            continue

                for k in self.particle_map.keys():
                    if k not in counter:
                        counter[k] = 0

                counter = {k: counter[k] for k in sorted(counter.keys())}
                self.run_stats_list[i] = counter

            for d in self.run_stats_list:
                arr = []
                for k, v in d.items():
                    arr += [self.particle_map[k]] * v
                self.arrs.append(arr)

        @staticmethod
        def is_arr_close_dist(arr1, arr2, threshold=0.05):
            ks_stat, ks_p_value = ks_2samp(arr1, arr2)
            return ks_p_value > threshold

        def make_buckets(self, threshold=0.05):
            if len(self.run_stats_list):
                stats = copy.deepcopy(self.run_stats_list[0])
                self.buckets = [[stats, set([0])]]

            add_new = True
            for i in range(1, len(self.arrs)):
                for j in range(len(self.buckets)):
                    arr1 = self.arrs[i]
                    arr2 = []
                    for k, v in self.buckets[j][0].items():
                        arr2 += [self.particle_map[k]] * v
                    if self.is_arr_close_dist(arr1, arr2, threshold):
                        self.buckets[j][0] = {k: self.buckets[j][0][k] + self.run_stats_list[i][k] for k in
                                              self.run_stats_list[i].keys()}
                        self.buckets[j][1].add(i)
                        add_new = False
                        break
                    else:
                        add_new = True

                if add_new:
                    stats = copy.deepcopy(self.run_stats_list[i])
                    self.buckets.append([stats, set([i])])

        def random_split_list(self, my_list, ks=[0.6, 0.2, 0.2]):
            random.shuffle(my_list)
            train = round(ks[0] * len(my_list))
            test1 = round(ks[1] * len(my_list))
            test2 = round(ks[2] * len(my_list))

            train_set = my_list[:train]
            test_set1 = my_list[train:train + test1]
            test_set2 = my_list[train + test1:train + test1 + test2]
            test_set3 = my_list[train + test1 + test2:]

            return train_set, test_set1, test_set2, test_set3

        def id2arr(self, ids):
            arr = []
            for i in ids:
                arr = arr + self.arrs[i]
            return arr

        def generate_datasets(self, ks=[0.6, 0.2, 0.2]):
            train_dt = []
            test_dt1 = []
            test_dt2 = []
            test_dt3 = []
            for bucket in self.buckets:
                train_set, test_set1, test_set2, test_set3 = self.random_split_list(list(bucket[1]), ks)
                train_dt = train_dt + train_set
                test_dt1 = test_dt1 + test_set1
                test_dt2 = test_dt2 + test_set2
                test_dt3 = test_dt3 + test_set3

            train_dataset = [self.tomograms[i] for i in train_dt]
            test_dataset1 = [self.tomograms[i] for i in test_dt1]
            test_dataset2 = [self.tomograms[i] for i in test_dt2]
            test_dataset3 = [self.tomograms[i] for i in test_dt3]

            return train_dataset, test_dataset1, test_dataset2, test_dataset3

    datasets = SplitDataset(root, specific_user_id, specific_session_id)
    datasets.make_buckets()

    train_dataset, test_dataset1, test_dataset2, test_dataset3 = datasets.generate_datasets(ks_values)

    output_data = {
        'train_dataset': train_dataset,
        'test_dataset1': test_dataset1,
        'test_dataset2': test_dataset2,
        'test_dataset3': test_dataset3,
        'run_stats_list': datasets.run_stats_list,  # Add this line to output the particle count stats
        'buckets': [
            {
                'stats': {k: int(v) for k, v in bucket[0].items()},
                'indices': list(bucket[1])
            }
            for bucket in datasets.buckets
        ]
    }

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

setup(
    group="copick",
    name="split-dataset",
    version="0.0.5",
    title="Split Dataset for Training and Testing",
    description="A solution that splits datasets into training and test sets, ensuring distributions are preserved.",
    solution_creators=["Kevin Zhao and Kyle Harrington"],
    tags=["data analysis", "dataset splitting", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "ks_values", "type": "string", "required": True, "description": "Comma-separated list of split ratios for train, test1, test2, and test3."},
        {"name": "output_json_path", "type": "string", "required": True, "description": "Path to the output JSON file."},
        {"name": "user_id", "type": "string", "required": True, "description": "User ID to filter picks."},
        {"name": "session_id", "type": "string", "required": True, "description": "Session ID to filter picks."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
