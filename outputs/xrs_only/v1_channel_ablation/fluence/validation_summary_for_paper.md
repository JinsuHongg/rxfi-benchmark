# Fluence channel ablation: validation results

## Study setup

- **Task:** Predict fluence with `compact_cnn_v1` using the V1 temporal split.
- **Input conditions:** all 13 channels as the reference, AIA 8, HMI 5, and six single-channel inputs.
- **Usable samples:** 70,699 train, 3,473 validation, and 41,910 test samples. The selected runs for all channel conditions report the same cohort counts.
- **Input preprocessing:** signed-log input transform followed by train-fitted channel z-score normalization.
- **Target preprocessing:** `asinh_zscore`, fitted from training targets only.
- **Training:** AdamW, quantile pinball objective at q05/q50/q95, 70 epoch maximum, cosine schedule with 10% warmup, early-stopping patience 7 after epoch 4, batch size 32.
- **Search grid:** learning rates `5e-7`, `1e-6`, and `2e-6`; weight decay `0.03`, `0.05`, and `0.1`; seeds 1, 2, and 3. This produced 27 runs per input condition.
- **Checkpoint and configuration selection:** each run's checkpoint was selected by minimum validation total pinball loss. For each input condition, the LR/WD pair was selected by the lowest mean validation q50 pinball loss across its three seeds.
- **Reported values:** mean ± sample standard deviation across the three selected seed runs. Errors, confidence intervals, and significance tests were not estimated.

## Results

| Input condition | Selected LR / WD | Total pinball ↓ | q50 pinball ↓ | MAE q50 ↓ | RMSE q50 ↓ | Spearman q50 ↑ | R² q50 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIA 8 | 2e-6 / 0.05 | 0.070637 ± 0.001608 | 0.124448 ± 0.003548 | 0.248896 ± 0.007096 | 0.436813 ± 0.003374 | 0.953231 ± 0.001312 | 0.803970 ± 0.003028 |
| 13-channel reference | 1e-6 / 0.03 | 0.072637 ± 0.001924 | 0.126251 ± 0.002064 | 0.252503 ± 0.004128 | 0.439076 ± 0.006642 | 0.949815 ± 0.003109 | 0.801911 ± 0.005968 |
| Single AIA 1600 | 5e-7 / 0.1 | 0.081842 ± 0.001145 | 0.154266 ± 0.002631 | 0.308531 ± 0.005262 | 0.483173 ± 0.007481 | 0.928924 ± 0.002884 | 0.760122 ± 0.007444 |
| Single AIA 211 | 5e-7 / 0.1 | 0.084820 ± 0.002755 | 0.155411 ± 0.003535 | 0.310822 ± 0.007071 | 0.487176 ± 0.011196 | 0.927081 ± 0.002981 | 0.756084 ± 0.011134 |
| Single AIA 193 | 1e-6 / 0.1 | 0.086705 ± 0.003714 | 0.156687 ± 0.001118 | 0.313374 ± 0.002235 | 0.493847 ± 0.014675 | 0.928300 ± 0.002375 | 0.749299 ± 0.014884 |
| HMI 5 | 2e-6 / 0.03 | 0.088276 ± 0.001243 | 0.157917 ± 0.000669 | 0.315835 ± 0.001339 | 0.523802 ± 0.007543 | 0.918537 ± 0.003008 | 0.718090 ± 0.008088 |
| Single HMI M | 2e-6 / 0.03 | 0.090851 ± 0.001541 | 0.162675 ± 0.004628 | 0.325350 ± 0.009256 | 0.531057 ± 0.009809 | 0.914149 ± 0.002724 | 0.710202 ± 0.010654 |
| Single AIA 131 | 5e-7 / 0.1 | 0.090380 ± 0.000229 | 0.165101 ± 0.001557 | 0.330202 ± 0.003113 | 0.534688 ± 0.003853 | 0.912615 ± 0.000302 | 0.706282 ± 0.004241 |
| Single AIA 171 | 2e-6 / 0.03 | 0.099326 ± 0.001185 | 0.177530 ± 0.001518 | 0.355060 ± 0.003035 | 0.566809 ± 0.001097 | 0.910949 ± 0.003233 | 0.669942 ± 0.001278 |

MAE, RMSE, and R² are computed in the `asinh_zscore` target space. Spearman is computed from q50 predictions. Total pinball is the mean over q05, q50, and q95. MSE, interval coverage, calibration fractions, interval width, and quantile crossing metrics remain available in the per-run `training_log.csv` files.

## Interpretation and limits

On this validation split, AIA 8 has slightly lower pinball losses than the 13-channel reference, with similar Spearman and transformed-space R². The observed differences are descriptive; only three seeds were used, and no inferential test was performed. Single AIA 1600 is the strongest single-channel condition by validation q50 pinball loss, while all tested single-channel conditions have higher validation q50 pinball loss than AIA 8.

These are validation results used to select LR/WD settings and checkpoints. They do not establish final test performance or statistical superiority. The 13-channel fluence test results were previously evaluated; the planned channel comparison will reuse those exact baseline checkpoint results and evaluate the other 24 selected checkpoints once each. Because the baseline test results have already been reviewed, the complete channel comparison should be described as exploratory rather than as an untouched confirmatory evaluation.

## Source artifacts

- Per-run metrics and training histories: `outputs/xrs_only/v1_channel_ablation/fluence/<condition>/<run>/summary_metrics.csv` and `training_log.csv`.
- Selected run configurations and cohort sizes: the corresponding `resolved_config.yaml` and `cohort_summary.csv` files.
- 13-channel reference runs: `outputs/xrs_only/v1_sweeps/fluence/compact_cnn/lrwd_grid/`.
- Existing 13-channel test records: `outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv`.
