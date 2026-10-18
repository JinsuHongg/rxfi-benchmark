#!/usr/bin/env bash
#SBATCH --job-name=surya_rechunk
#SBATCH --partition=qCPU24
#SBATCH --output=surya_rechunk_%j.out
#SBATCH --error=surya_rechunk_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"

SOURCE_ZARR_PATH="${SOURCE_ZARR_PATH:-/scratch/users/jhong36/surya-bench-224-interarea.zarr}"
DESTINATION_ZARR_PATH="${DESTINATION_ZARR_PATH:-/scratch/users/jhong36/surya-bench-224-interarea-sample-chunks.zarr}"
STAGING_ZARR_PATH="${DESTINATION_ZARR_PATH}.incomplete.${SLURM_JOB_ID:-manual}"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset}"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "source=$SOURCE_ZARR_PATH destination=$DESTINATION_ZARR_PATH"

srun python scripts/data/rechunk_surya_zarr.py \
  --source "$SOURCE_ZARR_PATH" \
  --destination "$DESTINATION_ZARR_PATH" \
  --staging "$STAGING_ZARR_PATH"
