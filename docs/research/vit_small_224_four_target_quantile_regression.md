# Cluster-Ready Four-Target ViT-Small Quantile Regression

QR-1 through QR-4 use identical SuryaBench 13-channel 224x224 inputs, frozen membership, ViT-Small/16 (`pretrained=false`), q05/q50/q95 outputs, pinball loss, AdamW (`1e-4`, weight decay `0.01`), ten epochs, seed 0, and minimum validation pinball loss checkpoint selection. Only target, transform, experiment name, and output directory differ.

| Run | Raw target | Transform | Inverse |
| --- | --- | --- | --- |
| QR-1 | `max_peak_flux` | `log10(max_peak_flux / 1e-8)` | `1e-8 * 10^z` |
| QR-2 | `cumulative_peak_flux` | `log10(1 + cumulative_peak_flux / 1e-8)` | `1e-8 * (10^z - 1)` |
| QR-3 | `max_rxfi` | `log10(1 + max_rxfi)` | `10^z - 1` |
| QR-4 | `cumulative_rxfi` | `log10(1 + cumulative_rxfi)` | `10^z - 1` |

QR-1's scaled transform intentionally supersedes its earlier unscaled `log10(max_peak_flux)` representation; raw targets are unchanged. The shared export schema is `split, original_row_index, timestamp, target_name, target_raw, target_transformed, max_flare_class, q05, q50, q95, q05_raw, q50_raw, q95_raw`.

The generic launcher requires explicit GPU assignments and prevents accidental four-process sharing: `QR1_GPU=0 QR2_GPU=1 QR3_GPU=2 QR4_GPU=3 QR_BATCH_SIZE=64 QR_PRECISION=bf16-mixed QR_NUM_WORKERS=8 bash scripts/run_all_qr_experiments.sh`. It accepts `32`, `16-mixed`, or `bf16-mixed`; the learning rate remains fixed unless explicitly changed. `scripts/slurm/run_qr_array.sbatch` is a scheduler-neutral four-index template; set the local partition/account/time/memory before submission.

Use `python scripts/benchmark_qr_batch_size.py --config configs/vit_small_224_max_peak_flux_qr.yaml --batch-sizes 16 32 64 128 --precision bf16-mixed` to measure throughput and VRAM manually. No script selects a batch size automatically. After training, export best-checkpoint predictions for all four targets and apply the same fixed OCQR calibration/evaluation procedure.
