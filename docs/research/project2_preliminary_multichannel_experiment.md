# Project 2 preliminary multichannel experiment wave

## Scientific question

Does multichannel SDO input improve 24-hour flare prediction compared with physically motivated single-channel baselines?

## Input configurations

The canonical Zarr order is `aia94, aia131, aia171, aia193, aia211, aia304, aia335, aia1600, hmi_m, hmi_bx, hmi_by, hmi_bz, hmi_v`. The preliminary conditions are `hmi_m` (index 8; line-of-sight photospheric magnetic field), `aia131` (index 1; hot/flaring coronal plasma), `aia193` (index 3; broader coronal thermal/structural response), and `all13` (indices 0–12; combined multimodal information). They are physically motivated baselines, not claims that one channel universally represents an instrument.

## Tasks and controls

Each input condition is prepared for NOAA-derived 24-hour `max_flare_class` classification and cumulative-peak-flux QR with `log10(1 + F / 1e-8)`, q05/q50/q95, and validation-pinball checkpoint selection. ViT-Small/16, 224x224, no pretrained weights, frozen splits, optimizer, LR, weight decay, epochs, and within-family seed are fixed. Per-sample, per-selected-channel z-score normalization is applied after channel slicing, avoiding test-derived statistics.

## Cohort policy and scope

The training scripts record channel-selected availability and cohort counts before training. Final comparisons must use the common test cohort across all four channel conditions and separately disclose full-cohort sensitivity results. This wave does not test all AIA/HMI variants, architecture size, resolution, multiple seeds, or forecasting horizons.

## Execution

Use `scripts/project2/run_project2_preliminary.sh classification`, `regression`, or `all` only with explicit GPU mappings. The Slurm array template supports `TASK_FAMILY=classification` or `regression`. Batch size, precision (`32`, `16-mixed`, `bf16-mixed`), workers, and gradient accumulation are launch-time overrides; LR is never auto-scaled.
