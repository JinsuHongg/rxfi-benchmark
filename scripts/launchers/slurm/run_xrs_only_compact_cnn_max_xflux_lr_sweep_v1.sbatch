#!/usr/bin/env bash
#SBATCH --job-name=xrs_compact_xflux_v1
#SBATCH --partition=qGPU24
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_only_compact_cnn_max_xflux_lr_v1/%x_%A_%a.out
#SBATCH --error=logs/xrs_only_compact_cnn_max_xflux_lr_v1/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_only_compact_cnn_max_xflux_lr_v1

learning_rates=(7e-7 1e-6 2e-6 5e-6 7e-6 1e-5)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
learning_rate="${learning_rates[$task_id]}"
lr_label="${learning_rate//./p}"
output_dir="outputs/xrs_only/v1_sweeps/max_xflux/compact_cnn/lr${lr_label}_wd0p05_seed0_${SLURM_JOB_ID}_${task_id}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun --cpu-bind=none python -m scripts.training.train_vit_quantile_regression \
  --config configs/xrs_only/compact_cnn_224_max_xflux_qr_v1_seed0.yaml \
  --learning-rate "$learning_rate" \
  --weight-decay 0.05 \
  --seed 0 \
  --output-dir "$output_dir" \
  --wandb-group compact-cnn-max-xflux-lr-v1 \
  --wandb-job-type lr-screen \
  --skip-sample-predictions \
  --skip-test \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}" \
  --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}" \
  --epochs 70
