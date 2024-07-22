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
  - "numpy<2"
  - scipy
  - scikit-image
  - joblib
  - scikit-learn==1.3.2
  - pip:
    - album
    - "git+https://github.com/uermel/copick.git"
"""

def run():
    import json
    import subprocess
    from copick.impl.filesystem import CopickRootFSSpec

    # Fetch arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    album_solution_name = args.album_solution_name
    slurm_partition = args.slurm_partition
    slurm_time = args.slurm_time
    slurm_memory = args.slurm_memory
    slurm_cpus_per_task = args.slurm_cpus_per_task
    slurm_gpus = args.slurm_gpus
    slurm_module_commands = args.slurm_module_commands
    extra_args = args.extra_args
    submit_job = args.submit_job

    # Load Copick configuration
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = CopickRootFSSpec.from_file(copick_config_path)
    print("Copick root loaded successfully")

    # Get all run names in the Copick project
    run_names = [run.name for run in root.runs]
    num_runs = len(run_names)

    # Convert run names to a single string to avoid syntax issues in shell script
    run_names_str = "\n".join(run_names)

    # Construct the Slurm job script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=album_job_array
#SBATCH --output=album_job_%A_%a.out
#SBATCH --error=album_job_%A_%a.err
#SBATCH --array=0-{num_runs - 1}
#SBATCH --time={slurm_time}
#SBATCH --mem={slurm_memory}
#SBATCH --cpus-per-task={slurm_cpus_per_task}
#SBATCH --gpus={slurm_gpus}
"""

    if slurm_partition:
        slurm_script += f"#SBATCH --partition={slurm_partition}\n"

    if slurm_module_commands:
        slurm_script += f"\n# Load modules\n{slurm_module_commands}\n"

    slurm_script += f"""
# Activate micromamba environment
eval "$(micromamba shell hook --shell=bash)"

run_names=({json.dumps(run_names)})
run_name=${{run_names[SLURM_ARRAY_TASK_ID]}}

export MAMBA_CACHE_DIR=/hpc/mydata/kyle.harrington/micromamba_cache/dir_$SLURM_JOB_ID
    
micromamba run -n album album run {album_solution_name} --copick_config_path {copick_config_path} --run_name $run_name {extra_args}
"""

    slurm_script_file = "submit_album_job_array.sh"
    with open(slurm_script_file, 'w') as f:
        f.write(slurm_script)

    # Print or submit the job array to Slurm
    if submit_job:
        subprocess.run(["sbatch", slurm_script_file], check=True)
        print(f"Submitted job array for {num_runs} runs to Slurm using solution '{album_solution_name}'")
    else:
        print(f"Slurm submission command: sbatch {slurm_script_file}")
        with open(slurm_script_file, 'r') as f:
            print(f.read())

setup(
    group="copick",
    name="submit-album-job-array",
    version="0.0.9",
    title="Submit Album Job Array",
    description="Submit another album solution to Slurm as a job array by using the runs in a Copick project.",
    solution_creators=["Kyle Harrington"],
    tags=["slurm", "job array", "album", "copick"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {"name": "copick_config_path", "type": "string", "required": True, "description": "Path to the Copick configuration JSON file."},
        {"name": "album_solution_name", "type": "string", "required": True, "description": "Name of the album solution to run."},
        {"name": "slurm_partition", "type": "string", "required": False, "description": "Slurm partition to use."},
        {"name": "slurm_time", "type": "string", "required": False, "default": "24:00:00", "description": "Time limit for the Slurm job (e.g., 01:00:00 for 1 hour)."},
        {"name": "slurm_memory", "type": "string", "required": False, "default": "128G", "description": "Memory limit for the Slurm job (e.g., 125G for 125 GB)."},
        {"name": "slurm_cpus_per_task", "type": "integer", "required": False, "default": 24, "description": "Number of CPUs per Slurm task."},
        {"name": "slurm_gpus", "type": "integer", "required": False, "default": 0, "description": "Number of GPUs per Slurm task."},
        {"name": "slurm_module_commands", "type": "string", "required": False, "description": "Slurm module commands to load necessary modules (e.g., module load cuda/11.8.0_520.61.05\\nmodule load cudnn/8.8.1.3_cuda11)."},
        {"name": "extra_args", "type": "string", "required": False, "default": "", "description": "Additional arguments to pass to the album solution."},
        {"name": "submit_job", "type": "boolean", "required": False, "default": True, "description": "Whether to submit the job to Slurm or just print the submission command and script."}
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
