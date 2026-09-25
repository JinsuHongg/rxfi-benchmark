# Severe-Event Binary Metrics Design

## Goal

Produce a reproducible, non-training evaluation layer for severe 24-hour flare-event forecasts. It must add M+ and X+ binary metrics to completed Project 1 QR, Project 2 QR, and corrected Project 2 ordinal6 classification artifacts, without changing frozen splits, source predictions, checkpoints, or earlier result files.

## Scope and evidence boundary

QR sources are the completed prediction CSVs. Their verified columns include `split`, `original_row_index`, `timestamp`, `target_name`, `target_raw`, `target_transformed`, `max_flare_class`, transformed quantiles, and raw quantiles. All Project 1 four-target and Project 2 cumulative-peak-flux conditions contain validation, test, and leaky-validation files.

Corrected Project 2 classification artifacts contain a six-by-six TEST confusion matrix for each condition, but no row-level prediction/probability export and no validation or leaky-validation confusion matrix. Before collapsing each matrix, the evaluator must verify its CSV row and column labels against the corresponding saved `class_mapping.json`; it may not assume an ordering. Classification severe metrics are exact only for TEST, derived by collapsing the verified matrix; classification AUROC/AUPRC and non-test severe metrics are explicitly unavailable rather than estimated or filled.

The old 310-class classification artifacts are excluded.

## Binary definitions

The fixed ordinal mapping is `FQ=0, A=1, B=2, C=3, M=4, X=5`. M+ has positive classes M/X; X+ has positive class X. For direct corrected classification, a prediction is positive under the same collapse rule.

## Metric engine

One pure Python metric module will construct TP/FP/TN/FN and guarded scalar metrics. It will calculate precision, recall/POD, positive F1, negative F1, binary macro-F1, balanced accuracy, specificity, FPR, FNR, TSS, and HSS. Undefined ratios return `NaN` rather than an invented value. TSS/HSS will be calculated from the confusion matrix; standard binary metrics use sklearn where applicable. POD and positive-class recall are numerically identical in this binary formulation and are retained under both names for solar-forecast reporting, not treated as independent evidence. AUROC/AUPRC are threshold-independent QR q50 ranking metrics and are included only where a continuous QR q50 score is present and both classes occur.

## QR protocol

For each target/condition/event pair, q50 is the ranking score and validation is the only threshold-selection population. The candidate thresholds are all finite unique validation scores, with each rule predicting positive for `score >= threshold`. Select the maximum validation TSS; break ties by higher positive F1, then higher POD, then the numerically higher score. Freeze the chosen threshold and evaluate it unchanged on validation, test, and leaky-validation. TSS, HSS, F1, macro-F1, POD/recall, and balanced accuracy are threshold-dependent decision metrics and are always labelled as validation-selected frozen-threshold results. No TEST labels, TEST metrics, or TEST threshold sweeps may participate in threshold selection. The evaluator writes the full validation threshold sweep with the requested counts and decision metrics.

`max_peak_flux` additionally evaluates q50_raw against physical M1 (`1e-5 W/m^2`) and X1 (`1e-4 W/m^2`) cutoffs. This is valid because its raw target is the maximum 24-hour peak XRS-B flux. Cumulative peak flux is an event-sum and therefore does not receive a single-event GOES physical cutoff; RXFI targets have no direct GOES cutoff.

## Artifacts

All new generated results reside below `outputs/severe_event_binary_metrics/`, including classification TEST metrics, QR validation-threshold metrics, physical-threshold metrics, validation threshold curves, provenance/availability manifest, and compact Project 1/Project 2 TEST summary CSVs. Nothing under an existing experiment output directory is modified.

`docs/research/severe_event_binary_metrics.md` will explain definitions, provenance, protocol, results, and limitations; `docs/research/current_state.md` will point to this completed evaluation.

## Testing

Pytest tests will cover perfect, all-negative, all-positive, and manually checked confusion matrices; broad mapping and direct-class collapse; validation-only threshold selection; and the fact that altered test labels cannot change a selected threshold.

## Failure handling

Inputs are schema-checked and split labels, target names, finite scores, and matrix class order are validated. Missing source artifacts are recorded as unavailable in the manifest. The generated report distinguishes unavailable metrics from zero-valued metrics and states that test labels never select QR thresholds.
