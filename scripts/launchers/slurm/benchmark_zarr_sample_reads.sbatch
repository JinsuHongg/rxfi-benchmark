#!/usr/bin/env bash
#SBATCH --job-name=zarr_read_bench
#SBATCH --partition=qCPU24
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=zarr_read_bench_%j.out
#SBATCH --error=zarr_read_bench_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"

SOURCE_ZARR_PATH="${SOURCE_ZARR_PATH:-/scratch/users/jhong36/surya-bench-224-interarea.zarr}"
OPTIMIZED_ZARR_PATH="${OPTIMIZED_ZARR_PATH:-/scratch/users/jhong36/surya-bench-224-interarea-sample-chunks.zarr}"

srun python scripts/data/benchmark_zarr_sample_reads.py \
  --source "$SOURCE_ZARR_PATH" \
  --optimized "$OPTIMIZED_ZARR_PATH" \
  --year "${BENCHMARK_YEAR:-2010}" \
  --samples "${BENCHMARK_SAMPLES:-10}" \
  --repeats "${BENCHMARK_REPEATS:-3}"
