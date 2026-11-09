#!/usr/bin/env bash
#SBATCH --job-name=xrs_compact_fl_chan
#SBATCH --partition=qGPU24
#SBATCH --array=0-80%3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_only_fluence_channel_combinations_v1/%x_%A_%a.out
#SBATCH --error=logs/xrs_only_fluence_channel_combinations_v1/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_only_fluence_channel_combinations_v1

conditions=(aia211_aia193 aia211_aia1600 aia193_aia211_aia1600)
configs=(
  configs/xrs_only/compact_cnn_224_fluence_qr_v1_aia211_aia193.yaml
  configs/xrs_only/compact_cnn_224_fluence_qr_v1_aia211_aia1600.yaml
  configs/xrs_only/compact_cnn_224_fluence_qr_v1_aia193_aia211_aia1600.yaml
)
learning_rates=(5e-7 1e-6 2e-6)
weight_decays=(0.03 0.05 0.1)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition_index="$((task_id / 27))"
condition="${conditions[$condition_index]}"
config_path="${configs[$condition_index]}"
condition_task="$((task_id % 27))"
learning_rate="${learning_rates[$((condition_task / 9))]}"
weight_decay="${weight_decays[$(((condition_task / 3) % 3))]}"
seed="$((condition_task % 3 + 1))"
lr_label="${learning_rate//./p}"
wd_label="${weight_decay//./p}"
output_dir="outputs/xrs_only/v1_channel_ablation/fluence/${condition}/lr${lr_label}_wd${wd_label}_seed${seed}_${SLURM_JOB_ID}_${task_id}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"
export WANDB_ENTITY="${WANDB_ENTITY:-gsu-dmlab}"
export WANDB_PROJECT="rxfi-regression"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun --cpu-bind=none python -m scripts.training.train_vit_quantile_regression \
  --config "$config_path" \
  --learning-rate "$learning_rate" \
  --weight-decay "$weight_decay" \
  --seed "$seed" \
  --output-dir "$output_dir" \
  --wandb-group "compact-cnn-fluence-channel-v1-${condition}" \
  --wandb-job-type channel-combination-lr-wd-screen \
  --skip-sample-predictions \
  --skip-test \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}" \
  --gradient-accumulation-steps "${QR_GRADIENT_ACCUMULATION:-1}" \
  --epochs 70
