# Fluence channel ablation: test results

## Evaluation protocol

- **Task and split:** `fluence`, V1 temporal split, `compact_cnn_v1`.
- **Test cohort:** 41,910 usable samples for every input condition and seed.
- **Checkpoints:** use the LR/WD pair selected on validation for each input condition and the validation-selected checkpoint for each of seeds 1, 2, and 3. No checkpoint or hyperparameter was selected using test metrics.
- **Evaluation runs:** 24 new evaluations across AIA 8, HMI 5, and six single-channel inputs. The existing three 13-channel reference test evaluations were reused because their checkpoints match the selected validation configuration.
- **Execution:** Slurm job 4878579; all 24 tasks completed, with a maximum concurrency of three. The evaluator wrote 24 metric files and 24 sample-prediction files. W&B runs were created in `rxfi-regression-test`.
- **Aggregation:** mean ± sample standard deviation across three seeds. No inferential test or confidence interval was calculated.

## Results

| Input condition | Selected LR / WD | q50 pinball ↓ | Total pinball ↓ | MAE q50 ↓ | RMSE q50 ↓ | Spearman q50 ↑ | R² q50 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single HMI M | 2e-6 / 0.03 | **0.175490 ± 0.001138** | **0.092448 ± 0.000553** | **0.350980 ± 0.002276** | **0.522245 ± 0.002122** | 0.927441 ± 0.001364 | **0.829963 ± 0.001383** |
| AIA 8 | 2e-6 / 0.05 | 0.181911 ± 0.033263 | 0.111179 ± 0.022776 | 0.363823 ± 0.066526 | 0.541917 ± 0.077382 | **0.951049 ± 0.003232** | 0.814425 ± 0.054198 |
| 13-channel reference | 1e-6 / 0.03 | 0.186345 ± 0.011870 | 0.101479 ± 0.011038 | 0.372690 ± 0.023740 | 0.580168 ± 0.028503 | 0.950603 ± 0.001828 | 0.789818 ± 0.020838 |
| Single AIA 131 | 5e-7 / 0.1 | 0.192890 ± 0.013920 | 0.097718 ± 0.005579 | 0.385781 ± 0.027840 | 0.542408 ± 0.031479 | 0.932023 ± 0.001350 | 0.816170 ± 0.021613 |
| Single AIA 171 | 2e-6 / 0.03 | 0.199085 ± 0.002429 | 0.106176 ± 0.001530 | 0.398170 ± 0.004858 | 0.604477 ± 0.005656 | 0.909002 ± 0.000858 | 0.772189 ± 0.004273 |
| HMI 5 | 2e-6 / 0.03 | 0.210865 ± 0.012355 | 0.109102 ± 0.006663 | 0.421730 ± 0.024711 | 0.601400 ± 0.040421 | 0.919202 ± 0.003523 | 0.773837 ± 0.029730 |
| Single AIA 193 | 1e-6 / 0.1 | 0.485614 ± 0.034257 | 0.295730 ± 0.051049 | 0.971228 ± 0.068514 | 1.249061 ± 0.081584 | 0.911693 ± 0.001600 | 0.024583 ± 0.129013 |
| Single AIA 211 | 5e-7 / 0.1 | 0.557780 ± 0.027324 | 0.347271 ± 0.040269 | 1.115559 ± 0.054647 | 1.400404 ± 0.083576 | 0.909623 ± 0.004750 | -0.225536 ± 0.145712 |
| Single AIA 1600 | 5e-7 / 0.1 | 0.703939 ± 0.023242 | 0.770299 ± 0.071434 | 1.407878 ± 0.046484 | 1.792900 ± 0.051396 | 0.777785 ± 0.118492 | -1.005116 ± 0.115669 |

MAE, RMSE, MSE, pinball losses, and R² use the `asinh_zscore` target space. Spearman is computed from q50 predictions. Total pinball is the mean of q05, q50, and q95 pinball losses. The per-run CSVs also contain quantile calibration fractions, interval coverage and width, and quantile-crossing metrics.

## Interpretation and limits

Among these evaluated configurations, single HMI M has the lowest mean test q50 pinball loss, while AIA 8 has the highest mean Spearman correlation. AIA 8 and the 13-channel reference have similar test rank correlations and q50 pinball losses. These comparisons are descriptive across three seeds and are not evidence of statistical significance.

Single AIA 1600, AIA 193, and AIA 211 show a large validation-to-test deterioration in pinball loss. Their test R² values are approximately -1.01, 0.02, and -0.23, respectively. This is an unexpected channel-specific generalization result; the current evaluation does not establish its cause. Do not tune these models against this test result. Investigate the saved predictions and input distributions using a development split, or use a new independent evaluation set for any revised model.

The 13-channel fluence test results were produced and reviewed before this channel-ablation evaluation. Consequently, this combined comparison is exploratory and should not be described as an untouched confirmatory test. The test metrics must not be used to select new hyperparameters or checkpoints.

## Result artifacts

- Combined 27-run summary: `outputs/xrs_only/v1_channel_ablation_test/summary/test_metrics_summary.csv`.
- Per-run metrics: `outputs/xrs_only/v1_channel_ablation_test/metrics/`.
- Per-sample predictions: `outputs/xrs_only/v1_channel_ablation_test/predictions/`.
- Reused 13-channel reference metrics and predictions: `outputs/xrs_only/v1_test_evaluation/`.
- Validation selection and run details: `outputs/xrs_only/v1_channel_ablation/fluence/validation_summary_for_paper.md`.
