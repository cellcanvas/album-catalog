###album catalog: cellcanvas

from album.runner.api import setup, get_args
import csv
import subprocess

env_file = """
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import os
    from copick.impl.filesystem import CopickRootFSSpec

    args = get_args()
    copick_config_path = args.copick_config_path
    reference_user_id = args.reference_user_id
    reference_session_id = args.reference_session_id
    distance_multiplier = float(args.distance_threshold)
    beta = float(args.beta)
    output_directory = args.output_directory
    candidates_csv = args.candidates_csv
    weights = args.weights
    slurm_partition = args.slurm_partition
    slurm_time = args.slurm_time
    slurm_memory = args.slurm_memory
    slurm_cpus_per_task = args.slurm_cpus_per_task
    slurm_module_commands = args.slurm_module_commands
    submit_job = args.submit_job

    # Load Copick configuration
    root = CopickRootFSSpec.from_file(copick_config_path)

    # Read candidates from CSV
    user_session_pairs = []
    with open(candidates_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            user_id, session_id = row
            user_session_pairs.append((user_id, session_id))

    num_pairs = len(user_session_pairs)

    # Convert user_session_pairs to a single string to avoid syntax issues in shell script
    user_session_pairs_str = " ".join(f'"{user_id}:{session_id}"' for user_id, session_id in user_session_pairs)

    # Construct the Slurm job script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=album_compare_picks
#SBATCH --output={output_directory}/album_job_%A_%a.out
#SBATCH --error={output_directory}/album_job_%A_%a.err
#SBATCH --array=0-{num_pairs - 1}
#SBATCH --time={slurm_time}
#SBATCH --mem={slurm_memory}
#SBATCH --cpus-per-task={slurm_cpus_per_task}
"""

    if slurm_partition:
        slurm_script += f"#SBATCH --partition={slurm_partition}\n"

    if slurm_module_commands:
        slurm_script += f"\n# Load modules\n{slurm_module_commands}\n"

    slurm_script += f"""
# Activate micromamba environment
eval "$(micromamba shell hook --shell=bash)"

user_session_pairs=({user_session_pairs_str})
user_session_pair=${{user_session_pairs[$SLURM_ARRAY_TASK_ID]}}
IFS=':' read -r candidate_user_id candidate_session_id <<< "$user_session_pair"

micromamba_cmd="micromamba run -n album album run compare-picks --copick_config_path {copick_config_path} --reference_user_id {reference_user_id} --reference_session_id {reference_session_id} --candidate_user_id $candidate_user_id --candidate_session_id $candidate_session_id --distance_threshold {distance_multiplier} --beta {beta} --output_json {output_directory}/result_${{candidate_user_id}}_${{candidate_session_id}}.json --weights '{weights}'"
echo "Executing: $micromamba_cmd"
eval $micromamba_cmd
"""

    slurm_script_file = os.path.join(output_directory, "submit_album_compare_picks.sh")
    with open(slurm_script_file, 'w') as f:
        f.write(slurm_script)

    # Print or submit the job array to Slurm
    if submit_job:
        subprocess.run(["sbatch", slurm_script_file], check=True)
        print(f"Submitted job array for {num_pairs} user-session pairs to Slurm using solution 'compare-picks'")
    else:
        print(f"Slurm submission command: sbatch {slurm_script_file}")
        with open(slurm_script_file, 'r') as f:
            print(f.read())

setup(
    group="copick",
    name="bulk-evaluation",
    version="0.0.2",
    title="Compare All Picks from Different Users and Sessions",
    description="A solution that uses the compare-picks album solution to evaluate all user_id and session_id pairs listed in a CSV file, creating JSON output files for each pair in a specified directory and submitting jobs to Slurm.",
    solution_creators=["Kyle Harrington"],
    tags=["data analysis", "picks", "comparison", "copick", "slurm"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "reference_user_id", "type": "string", "required": True, "description": "User ID for the reference picks."},
        {"name": "reference_session_id", "type": "string", "required": True, "description": "Session ID for the reference picks."},
        {"name": "distance_threshold", "type": "float", "required": True, "description": "Distance threshold multiplier for matching points (e.g., 1.5x the radius as default)."},
        {"name": "beta", "type": "float", "required": True, "description": "Beta value for the F-beta score."},
        {"name": "output_directory", "type": "string", "required": True, "description": "Directory to save the output JSON files."},
        {"name": "candidates_csv", "type": "string", "required": True, "description": "Path to the CSV file containing user_id and session_id pairs to process."},
        {"name": "weights", "type": "string", "required": True, "description": "JSON string with weights for each particle type."},
        {"name": "slurm_partition", "type": "string", "required": False, "description": "Slurm partition to use."},
        {"name": "slurm_time", "type": "string", "required": False, "default": "24:00:00", "description": "Time limit for the Slurm job (e.g., 01:00:00 for 1 hour)."},
        {"name": "slurm_memory", "type": "string", "required": False, "default": "128G", "description": "Memory limit for the Slurm job (e.g., 125G for 125 GB)."},
        {"name": "slurm_cpus_per_task", "type": "integer", "required": False, "default": 24, "description": "Number of CPUs per Slurm task."},
        {"name": "slurm_module_commands", "type": "string", "required": False, "description": "Slurm module commands to load necessary modules (e.g., module load cuda/11.8.0_520.61.05\\nmodule load cudnn/8.8.1.3_cuda11)."},
        {"name": "submit_job", "type": "boolean", "required": False, "default": True, "description": "Whether to submit the job to Slurm or just print the submission command and script."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
