#!/usr/bin/env bash
#SBATCH --job-name=xrs_qr_test
#SBATCH --partition=qGPU24
#SBATCH --array=0-29%3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/xrs_only_final_test/%x_%A_%a.out
#SBATCH --error=logs/xrs_only_final_test/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/xrs_only_final_test

manifest="${XRS_TEST_MANIFEST:-configs/xrs_only/v1_test_checkpoints.csv}"
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
manifest_row="$(awk -F, -v index="$((task_index + 2))" 'NR == index { print $1 "\t" $2 "\t" $3 "\t" $4 }' "$manifest")"
if [[ -z "$manifest_row" ]]; then
  logger_error="No manifest row for array task ${task_index} in ${manifest}"
  printf '%s\n' "$logger_error" >&2
  exit 1
fi
IFS=$'\t' read -r expected_target expected_seed config_path checkpoint_path <<< "$manifest_row"
if [[ ! -f "$config_path" || ! -f "$checkpoint_path" ]]; then
  printf 'Missing config or checkpoint for task %s: %s %s\n' \
    "$task_index" "$config_path" "$checkpoint_path" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-rxfi-benchmark}"
export WANDB_ENTITY="${WANDB_ENTITY:-gsu-dmlab}"
export WANDB_PROJECT="rxfi-regression-test"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/training${PYTHONPATH:+:$PYTHONPATH}"

srun --cpu-bind=none python -m scripts.training.evaluate_xrs_only_qr \
  --config "$config_path" \
  --checkpoint "$checkpoint_path" \
  --output-root "${XRS_TEST_OUTPUT_ROOT:-outputs/xrs_only/v1_test_evaluation}" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group compact-cnn-v1-final-test \
  --expected-target "$expected_target" \
  --expected-seed "$expected_seed" \
  --task-index "$task_index" \
  --batch-size "${QR_BATCH_SIZE:-32}" \
  --precision "${QR_PRECISION:-32}" \
  --num-workers "${QR_NUM_WORKERS:-4}"
