#!/usr/bin/env bash
#SBATCH --job-name=xrs_vit_tiny_xflux_sweep
#SBATCH --partition=qGPU24
#SBATCH --array=0-8%3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_only_vit_tiny_max_xflux_lrwd/%x_%A_%a.out
#SBATCH --error=logs/xrs_only_vit_tiny_max_xflux_lrwd/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_only_vit_tiny_max_xflux_lrwd

learning_rates=(2e-6 5e-6 1e-5)
weight_decays=(0.03 0.05 0.1)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
learning_rate="${learning_rates[$((task_id / ${#weight_decays[@]}))]}"
weight_decay="${weight_decays[$((task_id % ${#weight_decays[@]}))]}"
lr_label="${learning_rate//./p}"
wd_label="${weight_decay//./p}"
output_dir="outputs/xrs_only/sweeps/max_xflux_vit_tiny_lr${lr_label}_wd${wd_label}_seed0_${SLURM_JOB_ID}_${task_id}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun --cpu-bind=none python -m scripts.training.train_vit_quantile_regression \
  --config configs/xrs_only/vit_tiny_224_max_xflux_qr_splitv2_lr1e-6_seed0.yaml \
  --learning-rate "$learning_rate" \
  --weight-decay "$weight_decay" \
  --seed 0 \
  --output-dir "$output_dir" \
  --wandb-group vit-tiny-max_xflux-lrwd-screen-v1 \
  --wandb-job-type lrwd-screen \
  --skip-sample-predictions \
  --skip-test \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}" \
  --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}" \
  --epochs "${QR_EPOCHS:-30}"
