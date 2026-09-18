# Project 2 preliminary multichannel experiment wave

## Scientific question

Does multichannel SDO input improve 24-hour flare prediction compared with physically motivated single-channel baselines?

## Input configurations

The canonical Zarr order is `aia94, aia131, aia171, aia193, aia211, aia304, aia335, aia1600, hmi_m, hmi_bx, hmi_by, hmi_bz, hmi_v`. The preliminary conditions are `hmi_m` (index 8; line-of-sight photospheric magnetic field), `aia131` (index 1; hot/flaring coronal plasma), `aia193` (index 3; broader coronal thermal/structural response), and `all13` (indices 0–12; combined multimodal information). They are physically motivated baselines, not claims that one channel universally represents an instrument.

## Tasks and controls

Each input condition uses NOAA-derived 24-hour targets: ordinal `max_flare_class` classification and cumulative-peak-flux QR with `log10(1 + F / 1e-8)`, q05/q50/q95, and validation-pinball checkpoint selection. ViT-Small/16, 224x224 resolution, no pretrained weights, frozen splits, optimizer, learning rate, weight decay, epochs, and within-family seed are fixed. Per-sample, per-selected-channel z-score normalization is applied after channel slicing, avoiding test-derived statistics.

## Cohort policy

All four channel conditions have the same usable cohorts: 45,034 train, 2,422 validation, 27,980 test, and 3,755 leaky-validation rows. Therefore the primary channel comparisons use the same 27,980-row test cohort; no common-cohort split was created or modified.

## Corrected classification results

The first classifier used 310 detailed NOAA magnitude strings as separate classes and is excluded from the ordinal-band comparison. The corrected runs use fixed `FQ/A/B/C/M/X` mapping. A has zero support in the current test cohort, so primary macro-F1 and balanced accuracy average only observed `FQ/B/C/M/X`; the saved six-class macro-F1, which includes zero-support A as F1=0, remains available for provenance.

| Input | Accuracy | Observed macro-F1 | Observed balanced accuracy | M recall | X recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| hmi_m | 0.518 | 0.374 | 0.433 | 0.016 | 0.000 |
| aia131 | 0.578 | 0.463 | 0.483 | 0.253 | 0.000 |
| aia193 | 0.536 | 0.424 | 0.456 | 0.141 | 0.000 |
| all13 | 0.530 | 0.405 | 0.426 | 0.244 | 0.010 |

AIA131 has the strongest observed-class macro-F1 and balanced accuracy. All13 has the only nonzero X recall (10/968, 1.03%), but its classification gains are not consistent: it is lower than AIA131 on macro-F1, balanced accuracy, and M recall. Direct X classification therefore remains a rare-class limitation under this one-seed setup.

## Cumulative-peak-flux QR results

| Input | Spearman | R² | MAE/IQR | M+ AUPRC | M+ AUROC | X+ AUPRC | X+ AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hmi_m | 0.806 | 0.642 | 0.306 | 0.637 | 0.842 | 0.086 | 0.777 |
| aia131 | 0.818 | 0.743 | 0.262 | 0.664 | 0.853 | 0.110 | 0.795 |
| aia193 | 0.753 | 0.620 | 0.336 | 0.606 | 0.808 | 0.073 | 0.759 |
| all13 | 0.857 | 0.766 | 0.237 | 0.722 | 0.878 | 0.121 | 0.824 |

All13 is strongest on every listed QR metric. Relative to the strongest single-channel baseline, AIA131, all13 improves Spearman by 0.039, R² by 0.024, M+ AUPRC by 0.057, and X+ AUPRC by 0.012, while reducing MAE/IQR by 0.025. M+/X+ prevalence is 0.307/0.0346, respectively. Thus, under the current ViT-Small 224x224 setup, the 13-channel input improves both cumulative-peak-flux predictability and M+/X+ ranking relative to each tested single-channel baseline.

## Integrated finding

The full 13-channel input shows the strongest regression-based performance across the tested input conditions. Corrected multiclass classification does not show the same uniform pattern: AIA131 leads observed-class macro-F1 and balanced accuracy, whereas all13 is the only configuration with nonzero but very small X recall. QR retains a continuous ranking signal for rare X+ windows; this is not evidence that QR is globally superior to direct multiclass classification, because these tasks use different objectives and metrics.

## Convergence and runtime

Corrected classification best validation macro-F1 occurs at epochs 2, 1, 1, and 3 for hmi_m, aia131, aia193, and all13; none is selected at final epoch 10. QR best validation pinball epochs are 3, 2, 2, and 3, also not final. This last-epoch diagnostic does not itself suggest undertraining. Logged epoch-time sums are 4.94–5.25 h for corrected classification and 9.51–9.72 h for QR, on NVIDIA A30 with batch size 32, BF16 mixed precision, four workers, and gradient accumulation 1; these sums exclude setup and final evaluation.

## Limitations and next experiment wave

This is one seed, ViT-Small only, 224x224 only, and a 24-hour horizon only. The A band has no observed examples, X remains rare, no statistical significance analysis was performed, and the single-channel baselines do not exhaust possible SDO channels or channel groups. The next wave should first add seeds to distinguish stable channel effects from seed variation, then prioritize shorter 8 h/12 h horizons and selected physically motivated channel groups. Higher resolution and additional targets are subsequent controlled extensions, not automatic follow-ons.

## Artifacts

The integrated tables and figures are written under `outputs/project2/`, including `table_project2_preliminary_summary.csv`, `table_classification_ordinal6_test.csv`, `table_qr_cumulative_peak_flux_test.csv`, and the four comparison figures. The original 310-class outputs remain audit evidence only.
