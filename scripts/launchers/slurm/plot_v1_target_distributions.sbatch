#!/usr/bin/env bash
#SBATCH --job-name=v1_target_dist
#SBATCH --partition=qCPU24
#SBATCH --output=logs/v1_target_distributions/%x_%j.out
#SBATCH --error=logs/v1_target_distributions/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"

TRAIN_CSV="$PWD/data/derived_noaa_g17_v222/train.csv"
STATS_JSON="$PWD/outputs/xrs_only/v1_test_evaluation/summary/train_target_stats.json"
ZARR_PATH=/scratch/users/jhong36/surya-bench-224-interarea.zarr
OUTPUT_DIR="$PWD/outputs/xrs_only/v1_target_distributions"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "train_csv=$TRAIN_CSV zarr_path=$ZARR_PATH output_dir=$OUTPUT_DIR"

srun python -m scripts.analysis.plot_v1_target_distributions \
  --train-csv "$TRAIN_CSV" \
  --zarr-path "$ZARR_PATH" \
  --stats-json "$STATS_JSON" \
  --output-dir "$OUTPUT_DIR"
