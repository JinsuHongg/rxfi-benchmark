# V1 XRS-only experiment package

## Scope

This package records the ten XRS-only regression targets and the CompactCNN
v1 experiments that use the frozen SuryaBench v1 temporal split. It contains
validation results for background-window screening and final test metrics for
all ten targets across three seeds. The leaky-validation split was not used
for training or checkpoint selection.

## V1 split

The local split files are stored in `data/derived_noaa_g17_v222/` and are not
tracked in Git. The counts below are CSV data rows, excluding each header.
Shares use 128,328 rows across all four splits.

| Split | Time period and rule | Rows | Share |
| --- | --- | ---: | ---: |
| Train | 2010-05-13–2019-12-31; historical training rows outside holdout windows | 74,760 | 58.26% |
| Validation | January 15–31 in 2011–2019 | 3,672 | 2.86% |
| Leaky validation | January 1–14 and February 1–14 in 2011–2019 | 6,048 | 4.71% |
| Test | All timestamps in 2020–2024 | 43,848 | 34.17% |

The checksums identify the target-enriched v1 split files used by the
experiments:

| File | SHA-256 |
| --- | --- |
| `train.csv` | `f7714579ef9e969d37dd7c173dee7c128f5e86eed2c8af2c71f4779905a4c9a8` |
| `validation.csv` | `e8962f0f7f8547c1bfcdab8b3f5af3514325970f56ae19842f5a5d4a133186db` |
| `leaky_validation.csv` | `dee2249796fc924d54027c0949caf445b44e9319a2301e4e96f0cf14c7350b23` |
| `test.csv` | `fc36c51c537ab22329442d697d7ed1349d1985a15cacc94f355e7f797f53a1f5` |

These are source-row counts. Image availability and finite target coverage
reduce the usable sample count; the test summary records that count for every
target and seed.

## Target and training protocol

The target formulas and NOAA XRS quality policy are documented in
[`xrs_only_target_definitions.md`](../research/xrs_only_target_definitions.md).
The inputs are 224 × 224 images with 13 AIA/HMI channels, transformed with
signed `log1p` and standardized using training-split channel statistics. Each
target uses its training-split `asinh_zscore` parameters.

The model is `compact_cnn_v1`. Training used AdamW, cosine warmup over 10% of
steps, quantiles 0.05/0.50/0.95, a 70-epoch maximum, and early-stopping
patience of 7. Checkpoints were selected using validation pinball loss. The
per-target learning rate, weight decay, best validation epoch, and test sample
count are recorded in the [test metrics summary](../../outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv).

The 2-hour versus 24-hour background-window screening is summarized in
[`background_window_validation_comparison.md`](../research/background_window_validation_comparison.md).
That comparison is exploratory and uses validation results only. Its summary
table contains the reported values; trial-level logs remain in experiment
storage and are not part of this package.

## Result files

- [Test metrics summary](../../outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv): one row per target and seed.
- [Input normalization statistics](../../outputs/input_normalization_stats/train_channel_stats.json): training-only statistics for the 13 image channels.
- [Target normalization statistics](../../outputs/xrs_only/v1_test_evaluation/summary/train_target_stats.json): sanitized v1 target statistics used by the selected checkpoints.

Sample-level predictions and W&B runtime files remain in the experiment
storage and are not part of this compact result package.

## Provenance note

Final test evaluation ran as Slurm array job `4836855`. The local split files
are not included; their checksums above allow an available copy to be
verified. The original SuryaBench dataset revision used when the v1 target
files were generated still needs to be recovered before claiming a
bit-for-bit regeneration path.
