#!/usr/bin/env bash
#SBATCH --job-name=regression_targets_current
#SBATCH --partition=qCPU24
#SBATCH --output=logs/target_generation/%x_%j.out
#SBATCH --error=logs/target_generation/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-flare_surya}"

REVISION=bf474bc1e7752bd529e9707650f9310db8693763
DERIVED_DIR="$PWD/data/derived_noaa_current"
OUTPUT_DIR="$PWD/outputs/regression_target_experiment"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "revision=$REVISION derived_dir=$DERIVED_DIR output_dir=$OUTPUT_DIR"

srun python -m scripts.data.generate_regression_targets \
  --revision "$REVISION" \
  --derived-dir "$DERIVED_DIR" \
  --output-dir "$OUTPUT_DIR"
