# V1 fluence channel exploration: test evaluation

## Protocol

- V1 temporal split; CompactCNN; fluence target; seeds 1, 2, and 3.
- Each row uses the checkpoint selected by validation full pinball loss; LR/WD were selected by mean best-checkpoint validation full pinball loss across seeds.
- Test cohort: 41,910 usable samples per condition and seed.
- Metrics are mean +/- sample standard deviation across three seeds; no inferential test was performed.
- Prior test results had already informed selection of the five new channel combinations, so this combined comparison is exploratory, not an untouched confirmatory evaluation.
- Nine prior prediction files were reused where the validation-selected checkpoint exactly matched; raw metrics were recomputed from their saved samples. Thirty-three test evaluations were newly run in Slurm job 4886878, with concurrency limited to three.

## Transformed target space

| Input condition | LR / WD | Full pinball | q50 pinball | q50 MAE | q50 RMSE | Spearman | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 13 channels | 2e-06 / 0.1 | 0.097146 +/- 0.008611 | 0.190408 +/- 0.028312 | 0.380817 +/- 0.056624 | 0.583667 +/- 0.083219 | 0.951531 +/- 0.002803 | 0.784738 +/- 0.058747 |
| AIA 8 | 5e-07 / 0.03 | 0.110911 +/- 0.026331 | 0.179695 +/- 0.029639 | 0.359390 +/- 0.059277 | 0.538480 +/- 0.070131 | 0.952061 +/- 0.002308 | 0.817185 +/- 0.048856 |
| HMI 5 | 2e-06 / 0.03 | 0.109102 +/- 0.006663 | 0.210865 +/- 0.012355 | 0.421730 +/- 0.024711 | 0.601400 +/- 0.040421 | 0.919202 +/- 0.003523 | 0.773837 +/- 0.029730 |
| Single AIA 131 | 5e-07 / 0.03 | 0.097835 +/- 0.005646 | 0.193172 +/- 0.014025 | 0.386344 +/- 0.028050 | 0.543221 +/- 0.031858 | 0.932006 +/- 0.001343 | 0.815610 +/- 0.021918 |
| Single AIA 1600 | 1e-06 / 0.1 | 0.749049 +/- 0.100792 | 0.681718 +/- 0.056835 | 1.363435 +/- 0.113669 | 1.743394 +/- 0.123656 | 0.778962 +/- 0.171265 | -0.901231 +/- 0.274300 |
| Single AIA 171 | 2e-06 / 0.05 | 0.106192 +/- 0.001423 | 0.199128 +/- 0.002243 | 0.398256 +/- 0.004485 | 0.604541 +/- 0.005223 | 0.908991 +/- 0.000863 | 0.772143 +/- 0.003947 |
| Single AIA 193 | 1e-06 / 0.1 | 0.295730 +/- 0.051049 | 0.485614 +/- 0.034257 | 0.971228 +/- 0.068514 | 1.249061 +/- 0.081584 | 0.911693 +/- 0.001600 | 0.024583 +/- 0.129013 |
| Single AIA 211 | 5e-07 / 0.1 | 0.347271 +/- 0.040269 | 0.557780 +/- 0.027324 | 1.115559 +/- 0.054647 | 1.400404 +/- 0.083576 | 0.909623 +/- 0.004750 | -0.225536 +/- 0.145712 |
| Single HMI M | 1e-06 / 0.03 | 0.093192 +/- 0.001875 | 0.177249 +/- 0.004056 | 0.354499 +/- 0.008113 | 0.523955 +/- 0.008394 | 0.927348 +/- 0.001197 | 0.828820 +/- 0.005509 |
| AIA 131 + AIA 1600 | 5e-07 / 0.03 | 0.104178 +/- 0.009054 | 0.185895 +/- 0.009329 | 0.371791 +/- 0.018658 | 0.559202 +/- 0.032207 | 0.923839 +/- 0.001773 | 0.804617 +/- 0.022800 |
| AIA 131 + AIA 193 | 2e-06 / 0.03 | 0.110572 +/- 0.029655 | 0.182463 +/- 0.022774 | 0.364926 +/- 0.045548 | 0.540583 +/- 0.061144 | 0.945332 +/- 0.001814 | 0.816261 +/- 0.042412 |
| AIA 131 + AIA 211 | 2e-06 / 0.03 | 0.096632 +/- 0.006575 | 0.171699 +/- 0.001283 | 0.343398 +/- 0.002565 | 0.485506 +/- 0.006987 | 0.936401 +/- 0.001447 | 0.853026 +/- 0.004217 |
| AIA 1600 + AIA 193 | 1e-06 / 0.03 | 0.124093 +/- 0.022952 | 0.221097 +/- 0.041207 | 0.442193 +/- 0.082414 | 0.635763 +/- 0.128656 | 0.906034 +/- 0.001091 | 0.741132 +/- 0.107945 |
| AIA 131 + AIA 1600 + AIA 193 | 5e-07 / 0.03 | 0.142621 +/- 0.026771 | 0.228924 +/- 0.021815 | 0.457847 +/- 0.043631 | 0.668791 +/- 0.053328 | 0.941743 +/- 0.002963 | 0.719968 +/- 0.043676 |

## Original fluence space

| Input condition | q50 MAE | q50 RMSE | Spearman | R² |
|---|---:|---:|---:|---:|
| All 13 channels | 0.002851084 +/- 0.00135 | 0.015600836 +/- 0.00775 | 0.951531 +/- 0.002803 | -20.621893 +/- 20.004047 |
| AIA 8 | 0.0012845738 +/- 0.000169 | 0.0056554507 +/- 0.00215 | 0.952061 +/- 0.002308 | -1.674260 +/- 1.747592 |
| HMI 5 | 0.0019392924 +/- 0.000265 | 0.0055459646 +/- 0.000918 | 0.919202 +/- 0.003523 | -1.389336 +/- 0.775834 |
| Single AIA 131 | 0.0016938741 +/- 0.000212 | 0.0063953977 +/- 0.00142 | 0.932006 +/- 0.001343 | -2.222233 +/- 1.372113 |
| Single AIA 1600 | 0.0022001154 +/- 5.42e-05 | 0.0041811677 +/- 4.36e-05 | 0.778962 +/- 0.171265 | -0.333782 +/- 0.027913 |
| Single AIA 171 | 0.0012279285 +/- 5.13e-06 | 0.0030966011 +/- 1.43e-05 | 0.908991 +/- 0.000863 | 0.268465 +/- 0.006772 |
| Single AIA 193 | 0.0091200337 +/- 0.00214 | 0.025240947 +/- 0.0106 | 0.911692 +/- 0.001600 | -53.345962 +/- 43.569094 |
| Single AIA 211 | 0.012101996 +/- 0.00329 | 0.033184235 +/- 0.0137 | 0.909623 +/- 0.004750 | -92.532506 +/- 68.318557 |
| Single HMI M | 0.0013387679 +/- 7.85e-05 | 0.0033973991 +/- 0.000204 | 0.927348 +/- 0.001197 | 0.117340 +/- 0.106269 |
| AIA 131 + AIA 1600 | 0.0013420487 +/- 0.000189 | 0.0062876861 +/- 0.00337 | 0.923839 +/- 0.001772 | -2.595246 +/- 3.667274 |
| AIA 131 + AIA 193 | 0.048849953 +/- 0.0817 | 8.699098 +/- 14.9 | 0.945332 +/- 0.001814 | -17121165.212755 +/- 29653304.612175 |
| AIA 131 + AIA 211 | 0.0012604111 +/- 7.94e-05 | 0.0068850086 +/- 0.00163 | 0.936401 +/- 0.001447 | -2.750909 +/- 1.737355 |
| AIA 1600 + AIA 193 | 0.0013557955 +/- 0.000151 | 0.0032405131 +/- 0.00021 | 0.906034 +/- 0.001091 | 0.196662 +/- 0.105476 |
| AIA 131 + AIA 1600 + AIA 193 | 0.97224162 +/- 1.64 | 182.63206 +/- 308 | 0.941743 +/- 0.002963 | -7379544939.508719 +/- 12776072880.508411 |

## Artifacts

- Per-seed metrics and prediction paths: `outputs/xrs_only/v1_channel_exploration_test/summary/test_metrics_by_seed.csv`.
- Mean and sample standard deviation across seeds: `outputs/xrs_only/v1_channel_exploration_test/summary/test_metrics_summary.csv`.
- New task metrics and predictions are under `outputs/xrs_only/v1_channel_exploration_test/`; reused prediction CSVs remain under `outputs/xrs_only/v1_channel_ablation_test/predictions/`.

## Original-space prediction outliers

Saved predictions contain a small number of very large q50 values after inverting the asinh transform. Across all three seeds, AIA 131 + AIA 193 has six predictions with absolute q50 above 1 and a maximum absolute q50 of about 5,269; AIA 131 + AIA 1600 + AIA 193 has five above 1 and a maximum about 109,792. The largest absolute observed test target is about 0.043. These extremes dominate original-space RMSE and R² for those conditions. This is consistent with the exponential growth of inverse asinh for large transformed predictions; the cause of the transformed prediction outliers is not established. See `prediction_range_diagnostics.csv` and the per-sample prediction files for details.
