#!/usr/bin/env bash
#SBATCH --job-name=input_stats_splitv2
#SBATCH --partition=qCPU24
#SBATCH --output=logs/input_stats_splitv2_%j.out
#SBATCH --error=logs/input_stats_splitv2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"

ZARR_PATH=/scratch/users/jhong36/surya-bench-224-interarea.zarr
INDEX_CSV="$PWD/data/derived_noaa_g17_v222_splitv2/train.csv"
OUTPUT="$PWD/outputs/input_normalization_stats/noaa_g17_v222_splitv2/train_channel_stats.json"
mkdir -p "$(dirname "$OUTPUT")"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "zarr_path=$ZARR_PATH index_csv=$INDEX_CSV output=$OUTPUT"

srun python scripts/training/compute_input_stats.py \
  --zarr-path "$ZARR_PATH" \
  --index-csv "$INDEX_CSV" \
  --output "$OUTPUT" \
  --allow-missing-timestamps
