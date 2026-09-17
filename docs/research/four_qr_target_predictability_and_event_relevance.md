# Four QR Target Predictability and Severe-Event Relevance

## Purpose

This controlled Project 1 comparison separates two questions: how predictably each continuous target can be learned from SDO observations, and how useful its predicted median score is for ranking conventional severe solar-flare windows. A target being easier to predict does not by itself imply that it is more useful for severe solar-event forecasting.

## Experimental controls

QR-1 through QR-4 use the same frozen splits, 13-channel SDO inputs, 224x224 resolution, ViT-Small/16 architecture, q05/q50/q95 quantiles, AdamW protocol, seed, and validation-pinball checkpoint rule. Target representation alone changes: maximum peak flux, cumulative peak flux, maximum RXFI, or cumulative RXFI. The analysis uses completed best-validation-checkpoint prediction exports only; it does not retrain models or run OCQR.

## Cohort and method

All four test exports contain 27,980 rows with unique `(split, original_row_index, timestamp)` keys. Their intersection is the same 27,980 rows; therefore the common-cohort and target-specific full-cohort learnability values are identical here. Detailed GOES labels (for example `M2.4`) are mapped to their existing conventional bands (`FQ`, `B`, `C`, `M`, `X`) for C+/M+/X+ outcomes; the score remains transformed-space q50.

Primary learnability metrics are transformed-space Spearman correlation, R-squared, and MAE divided by the true transformed-target IQR. Raw MAE/RMSE are retained for target-specific interpretation, not for cross-target ranking. Event relevance uses AUPRC as the primary metric because M+ and especially X+ are imbalanced; AUROC is secondary. These are continuous-score ranking analyses, not thresholded classifiers or conformal-coverage claims.

## Results

On the common test cohort, the two peak-flux targets show higher predictability than the RXFI targets. Maximum peak flux has Spearman 0.826, R² 0.775, and normalized MAE 0.244. Cumulative peak flux has the highest Spearman (0.859) with normalized MAE 0.247 and R² 0.739. Maximum and cumulative RXFI have lower rank correlation (0.509 and 0.576) and higher normalized MAE (0.481 and 0.402).

For severe-event relevance, cumulative peak flux has the strongest observed ranking results: M+ AUPRC 0.715 against a no-skill prevalence baseline of 0.307, and X+ AUPRC 0.148 against a 0.0346 baseline. Maximum peak flux follows at M+ AUPRC 0.655 and X+ AUPRC 0.093. Maximum RXFI reaches 0.613 and 0.108, respectively; cumulative RXFI reaches 0.581 and 0.095. Thus the results distinguish learnability from downstream relevance: cumulative peak flux is slightly less favorable than maximum peak flux on transformed R²/normalized error, yet gives stronger severe-event ranking in this controlled setup.

Tail results are supplementary and should be interpreted cautiously. At the true top 5%, Spearman is weak or negative for every target, indicating that ordinary test-set ordering performance does not carry cleanly into the rarest target tails.

## Quantile and convergence audit

Best validation epochs are 2 for maximum peak flux and 1 for each remaining target. None has its best epoch at the final (tenth) epoch; final validation pinball loss is higher than the selected best loss for every run, so this specific last-epoch diagnostic does not flag possible undertraining. Test q05 empirical frequencies are below nominal for all targets, while q95 frequencies range roughly 0.896–0.958. These are quantile-regression calibration diagnostics, not conformal coverage.

Logged training-epoch time totals are about 7.06 h (QR-1), 7.05 h (QR-2), 5.15 h (QR-3), and 5.16 h (QR-4). The available-image cohort is 44,962 for QR-1 and 45,034 for the other three, so simple sample availability does not explain the observed runtime difference; no causal explanation is inferred here.

## Artifacts

The reproducible tables, figures, artifact manifest, cohort accounting, tail metrics, and convergence/runtime audit are under `outputs/four_qr_target_analysis/`. The main tables are `table_learnability_common_cohort.csv`, `table_learnability_full_cohort.csv`, `table_event_relevance.csv`, and `table_qr_diagnostics.csv`.

## Next step

Finalize the Project 1 target-representation interpretation, freeze a conformal calibration partition, then run a consistent OCQR calibration/evaluation procedure for the selected targets (or all four, if the comparison remains central).
