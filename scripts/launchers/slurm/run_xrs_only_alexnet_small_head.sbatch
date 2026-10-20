#!/usr/bin/env bash
#SBATCH --job-name=xrs_alexnet_small_head_qr
#SBATCH --partition=qGPU24
#SBATCH --array=0-2%4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_only_alexnet_small_head/%x_%A_%a.out
#SBATCH --error=logs/xrs_only_alexnet_small_head/%x_%A_%a.err

set -euo pipefail

configs=(
  configs/xrs_only/alexnet_small_head_224_max_xflux_qr_splitv2_seed0.yaml
  configs/xrs_only/alexnet_small_head_224_max_xflux_qr_splitv2_seed1.yaml
  configs/xrs_only/alexnet_small_head_224_max_xflux_qr_splitv2_seed2.yaml
)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_only_alexnet_small_head
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun --cpu-bind=none python -m scripts.training.train_vit_quantile_regression \
  --config "${configs[$SLURM_ARRAY_TASK_ID]}" \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}" \
  --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}" \
  --epochs "${QR_EPOCHS:-30}"
