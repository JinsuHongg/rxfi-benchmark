#!/usr/bin/env bash
#SBATCH --job-name=repartition_splitv2
#SBATCH --partition=qCPU24
#SBATCH --output=logs/repartition_splitv2_%j.out
#SBATCH --error=logs/repartition_splitv2_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"

srun python scripts/data/repartition_regression_targets.py \
  --config configs/preprocessing/noaa_g17_v222_splitv2.json
