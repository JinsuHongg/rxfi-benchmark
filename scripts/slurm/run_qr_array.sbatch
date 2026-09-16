#!/usr/bin/env bash
# Set partition/account/time/memory for the local cluster before submitting.
#SBATCH --job-name=rxfi_qr
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qr/%x_%A_%a.out
#SBATCH --error=logs/qr/%x_%A_%a.err
set -euo pipefail
configs=(configs/vit_small_224_max_peak_flux_qr.yaml configs/vit_small_224_cumulative_peak_flux_qr.yaml configs/vit_small_224_max_rxfi_qr.yaml configs/vit_small_224_cumulative_rxfi_qr.yaml)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/qr
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX to the QR conda environment}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV_PREFIX"
python scripts/train_vit_quantile_regression.py --config "${configs[$SLURM_ARRAY_TASK_ID]}" --batch-size "${QR_BATCH_SIZE:-4}" --precision "${QR_PRECISION:-32}" --num-workers "${QR_NUM_WORKERS:-0}" --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}"
