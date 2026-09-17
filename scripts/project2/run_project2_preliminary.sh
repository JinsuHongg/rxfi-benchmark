#!/usr/bin/env bash
# Run only with explicit, distinct GPU mappings; use Slurm for scheduled execution.
set -euo pipefail
mode="${1:-}"; [[ "$mode" =~ ^(classification|regression|all)$ ]] || { echo "usage: $0 {classification|regression|all}" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"; BATCH="${P2_BATCH_SIZE:-32}"; PRECISION="${P2_PRECISION:-32}"; WORKERS="${P2_NUM_WORKERS:-4}"; ACCUM="${P2_GRADIENT_ACCUMULATION:-1}"
channels=(hmi_m aia131 aia193 all13)
launch() { local task="$1" script="$2"; local i config gpu; for i in "${!channels[@]}"; do config="configs/project2/vit_small_224_${channels[$i]}_${task}.yaml"; gpu_var="P2_$((i+1))_GPU"; gpu="${!gpu_var:-}"; [[ -n "$gpu" ]] || { echo "Set P2_1_GPU through P2_4_GPU" >&2; exit 2; }; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$script" --config "$config" --batch-size "$BATCH" --precision "$PRECISION" --num-workers "$WORKERS" --gradient-accumulation-steps "$ACCUM" & done; wait; }
[[ "$mode" == classification || "$mode" == all ]] && launch max_flare_class scripts/train_vit_classifier.py
[[ "$mode" == regression || "$mode" == all ]] && launch cumulative_peak_flux_qr scripts/train_vit_quantile_regression.py
