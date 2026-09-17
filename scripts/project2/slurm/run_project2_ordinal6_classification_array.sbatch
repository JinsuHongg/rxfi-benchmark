#!/usr/bin/env bash
#SBATCH --job-name=rxfi_p2_ord6
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
set -euo pipefail
channels=(hmi_m aia131 aia193 all13); channel="${channels[$SLURM_ARRAY_TASK_ID]}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX}"; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV_PREFIX"
python scripts/train_vit_classifier.py --config "configs/project2/vit_small_224_${channel}_max_flare_class_ordinal6.yaml" --batch-size "${P2_BATCH_SIZE:-32}" --precision "${P2_PRECISION:-bf16-mixed}" --num-workers "${P2_NUM_WORKERS:-4}" --gradient-accumulation-steps "${P2_GRADIENT_ACCUMULATION:-1}"
