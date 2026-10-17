#!/usr/bin/env bash
#SBATCH --job-name=xrs_qr
#SBATCH --partition=qGPU24
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_qr/%x_%A_%a.out
#SBATCH --error=logs/xrs_qr/%x_%A_%a.err

set -euo pipefail

configs=(
  configs/xrs_only/vit_small_224_max_xflux_qr.yaml
  configs/xrs_only/vit_small_224_fluence_qr.yaml
  configs/xrs_only/vit_small_224_max_background_deviation_2h_q10_qr.yaml
  configs/xrs_only/vit_small_224_total_background_deviation_2h_q10_qr.yaml
  configs/xrs_only/vit_small_224_max_background_deviation_2h_q50_qr.yaml
  configs/xrs_only/vit_small_224_total_background_deviation_2h_q50_qr.yaml
  configs/xrs_only/vit_small_224_max_background_deviation_24h_q10_qr.yaml
  configs/xrs_only/vit_small_224_total_background_deviation_24h_q10_qr.yaml
  configs/xrs_only/vit_small_224_max_background_deviation_24h_q50_qr.yaml
  configs/xrs_only/vit_small_224_total_background_deviation_24h_q50_qr.yaml
)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_qr
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX to the rxfi-benchmark QR environment}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun python scripts/training/train_vit_quantile_regression.py \
  --config "${configs[$SLURM_ARRAY_TASK_ID]}" \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}" \
  --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}"
