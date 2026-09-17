#!/usr/bin/env bash
#SBATCH --job-name=rxfi_p2
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
# Override partition/time/memory/CPUs at submission according to the local cluster.
set -euo pipefail
TASK_FAMILY="${TASK_FAMILY:?set TASK_FAMILY=classification or regression}"
channels=(hmi_m aia131 aia193 all13); channel="${channels[$SLURM_ARRAY_TASK_ID]}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX}"; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV_PREFIX"
case "$TASK_FAMILY" in
 classification) python scripts/train_vit_classifier.py --config "configs/project2/vit_small_224_${channel}_max_flare_class.yaml" --batch-size "${P2_BATCH_SIZE:-32}" --precision "${P2_PRECISION:-32}" --num-workers "${P2_NUM_WORKERS:-4}" --gradient-accumulation-steps "${P2_GRADIENT_ACCUMULATION:-1}" ;;
 regression) python scripts/train_vit_quantile_regression.py --config "configs/project2/vit_small_224_${channel}_cumulative_peak_flux_qr.yaml" --batch-size "${P2_BATCH_SIZE:-32}" --precision "${P2_PRECISION:-32}" --num-workers "${P2_NUM_WORKERS:-4}" --gradient-accumulation-steps "${P2_GRADIENT_ACCUMULATION:-1}" ;;
 *) echo 'TASK_FAMILY must be classification or regression' >&2; exit 2 ;;
esac
