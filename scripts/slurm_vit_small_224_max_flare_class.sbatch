#!/usr/bin/env bash
#SBATCH --job-name=rxfi_vit_small_224
#SBATCH --partition=qGPU24
#SBATCH --output=logs/vit_small_224_max_flare_class/%x_%j.out
#SBATCH --error=logs/vit_small_224_max_flare_class/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Submit from the repository root:
#   mkdir -p logs/vit_small_224_max_flare_class
#   sbatch scripts/slurm_vit_small_224_max_flare_class.sbatch
# Optionally set CONDA_ENV_PREFIX or OUTPUT_DIR at submission time.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/vit_small_224_max_flare_class.yaml}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/users/jhong36/miniforge3/envs/rxfi-benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vit_small_224_max_flare_class_slurm_${SLURM_JOB_ID:?missing SLURM_JOB_ID}}"

test -f "$CONFIG" || {
  echo "Missing config: $CONFIG" >&2
  exit 2
}
test -d "$CONDA_ENV_PREFIX" || {
  echo "Missing Conda environment: $CONDA_ENV_PREFIX" >&2
  exit 2
}
test ! -e "$OUTPUT_DIR" || {
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 3
}
mkdir -p "$(dirname "$OUTPUT_DIR")" logs/vit_small_224_max_flare_class

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID} cwd=$PWD"
echo "config=$CONFIG output=$OUTPUT_DIR python=$(command -v python) conda_env=${CONDA_PREFIX:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch
print({"cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(), "torch": torch.__version__, "torch_cuda": torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is mandatory; refusing CPU fallback")
print({"gpu": torch.cuda.get_device_name(0)})
PY

# Preserve the tracked base configuration and record this job's output path in a
# job-local copy. The training script writes all metadata beside the outputs.
mkdir "$OUTPUT_DIR"
JOB_CONFIG="$OUTPUT_DIR/slurm_resolved_config.yaml"
python - "$CONFIG" "$JOB_CONFIG" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination, output_dir = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["output_dir"] = str(output_dir)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

python scripts/train_vit_classifier.py --config "$JOB_CONFIG"
