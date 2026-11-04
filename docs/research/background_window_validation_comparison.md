# CompactCNN background-window validation comparison

## Scope

This is an exploratory comparison of the 2-hour and 24-hour background
deviation targets in the CompactCNN v1 sweep on the v1 split. It records
validation results only; the test split was not used.

Each target had 27 screening trials. A trial checkpoint was selected by the
validation pinball loss averaged over model quantiles 0.05, 0.50, and 0.95.
The single-run rows below identify the trial with the lowest validation q50
pinball loss among those selected checkpoints. Target suffixes `q10` and
`q50` describe the target's background statistic; the reported q50 metrics
refer to the model's median prediction.

MAE and MSE are in the `asinh_zscore` standardized target space. Values are
rounded to four decimal places. Pearson r was not stored by the screening runs.

## Best single-run validation results

| Target | q50 pinball | R² | Spearman r | MAE | MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| max background deviation 2h q10 | 0.2244 | 0.4490 | 0.7528 | 0.4488 | 0.4755 |
| max background deviation 2h q50 | 0.2286 | 0.4188 | 0.7382 | 0.4571 | 0.5052 |
| max background deviation 24h q10 | 0.2197 | 0.4959 | 0.7807 | 0.4393 | 0.4340 |
| max background deviation 24h q50 | 0.2231 | 0.4864 | 0.7815 | 0.4461 | 0.4435 |
| total background deviation 2h q10 | 0.2861 | 0.1397 | 0.4909 | 0.5723 | 0.9149 |
| total background deviation 2h q50 | 0.2986 | 0.1111 | 0.4569 | 0.5971 | 0.9401 |
| total background deviation 24h q10 | 0.2380 | 0.4879 | 0.7002 | 0.4761 | 0.5199 |
| total background deviation 24h q50 | 0.2829 | 0.3168 | 0.5493 | 0.5658 | 0.7300 |

## Three-seed validation summary

For each target, the lowest mean q50 pinball across the three seeds was
selected from the nine learning-rate/weight-decay combinations. The values
below are mean plus or minus sample standard deviation across those seeds.

| Target | 2h mean q50 pinball | 24h mean q50 pinball |
| --- | ---: | ---: |
| Maximum background deviation q10 | 0.2272 ± 0.0008 | 0.2205 ± 0.0007 |
| Maximum background deviation q50 | 0.2312 ± 0.0008 | 0.2241 ± 0.0016 |
| Total background deviation q10 | 0.2865 ± 0.0005 | 0.2401 ± 0.0020 |
| Total background deviation q50 | 0.3013 ± 0.0019 | 0.2847 ± 0.0021 |

## Observation and limits

Both 24-hour targets have lower q50 pinball loss than their corresponding
2-hour targets in both background-deviation families. The best single-run
results also show higher R² and lower standardized MAE and MSE for the
24-hour conditions. Spearman r is higher for both maximum-background
comparisons and both total-background comparisons.

These are descriptive validation results from hyperparameter screening. The
minimum validation scores were selected from 27 trials per target, so they
may be optimistic. No significance test was performed, and these results do
not establish test-set performance or generalization beyond this split.
The summary values are sufficient for this comparison; trial-level training
logs remain in experiment storage and are not required for the proposal.
