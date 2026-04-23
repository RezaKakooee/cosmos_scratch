#!/bin/bash
#SBATCH --job-name=cosmos_overfit
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Load proxy for outbound access (HuggingFace) on compute nodes
PROJECT_ROOT_PROXY="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -f "$PROJECT_ROOT_PROXY/ops/set_proxy.sh" ]; then
    source "$PROJECT_ROOT_PROXY/ops/set_proxy.sh"
fi
unset PROJECT_ROOT_PROXY

py_script="scripts/overfit_one_batch.py"
config="configs/overfit_continuous.yaml"

run_job() {
    project_dir="/home2/reza/cosmos_scratch"

    echo "Printing SLURM directives:"
    awk '/^#SBATCH/ {print}' $0

    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    nvidia-smi

    echo "project_dir : $project_dir"
    echo "py_script   : $py_script"
    echo "config      : $config"

    echo "Activating conda environment: llm"
    eval "$(conda shell.bash hook)"
    conda activate llm

    cd "$project_dir"

    if [ -f .env ]; then
        set -a && source .env && set +a
        echo "loaded .env"
    fi

    python "$py_script" --config "$config"

    echo "Job completed. Output saved to ${output_file}"
}

output_dir="/home2/reza/cosmos_scratch/local_storage/logs"
current_date=$(date +%Y%m%d_%H%M%S)
job_id=${SLURM_JOB_ID}
output_file="${output_dir}/cosmos__${current_date}__${job_id}__overfit.out"

mkdir -p "$output_dir"

run_job > "${output_file}" 2>&1
