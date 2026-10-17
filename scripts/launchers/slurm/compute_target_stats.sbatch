#!/usr/bin/env bash
#SBATCH --job-name=target_stats
#SBATCH --partition=qCPU24
#SBATCH --output=logs/target_stats/%x_%j.out
#SBATCH --error=logs/target_stats/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-flare_surya}"

ZARR_PATH=/scratch/users/jhong36/surya-bench-224-interarea.zarr
TARGET_CSV="$PWD/data/derived/train.csv"
OUTPUT="$PWD/outputs/target_normalization_stats/train_target_stats.json"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "zarr_path=$ZARR_PATH target_csv=$TARGET_CSV output=$OUTPUT"

srun python scripts/training/compute_target_stats.py \
  --zarr-path "$ZARR_PATH" \
  --training-targets "$TARGET_CSV" \
  --output "$OUTPUT"
