#!/usr/bin/env bash
# Launch the four QR targets concurrently only when explicit GPU mappings exist.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
QR_BATCH_SIZE="${QR_BATCH_SIZE:-4}"
QR_PRECISION="${QR_PRECISION:-32}"
QR_NUM_WORKERS="${QR_NUM_WORKERS:-0}"
QR_GRADIENT_ACCUMULATION="${QR_GRADIENT_ACCUMULATION:-1}"
configs=(
  configs/vit_small_224_max_peak_flux_qr.yaml
  configs/vit_small_224_cumulative_peak_flux_qr.yaml
  configs/vit_small_224_max_rxfi_qr.yaml
  configs/vit_small_224_cumulative_rxfi_qr.yaml
)
gpus=("${QR1_GPU:-}" "${QR2_GPU:-}" "${QR3_GPU:-}" "${QR4_GPU:-}")

for gpu in "${gpus[@]}"; do
  [[ -n "$gpu" ]] || { echo "Set QR1_GPU through QR4_GPU explicitly; refusing shared-GPU launch." >&2; exit 2; }
done
if [[ "${QR_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -eq 4 ]] || { echo "GPU mappings must be distinct (or set QR_ALLOW_SHARED_GPU=1)." >&2; exit 2; }
fi
outputs=()
for config in "${configs[@]}"; do
  outputs+=("$($PYTHON_BIN -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")")
done
[[ "$(printf '%s\n' "${outputs[@]}" | sort -u | wc -l)" -eq 4 ]] || { echo "QR configs share an output directory." >&2; exit 2; }

for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON_BIN" scripts/train_vit_quantile_regression.py --config "${configs[$index]}" --batch-size "$QR_BATCH_SIZE" --precision "$QR_PRECISION" --num-workers "$QR_NUM_WORKERS" --gradient-accumulation-steps "$QR_GRADIENT_ACCUMULATION" &
done
wait
