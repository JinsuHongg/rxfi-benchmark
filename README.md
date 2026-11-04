# rxfi-benchmark

This repository supports research on how the definition of a solar-flare
forecasting target affects its learnability and operational relevance. It
includes reproducible target definitions, data preparation, and CompactCNN v1
regression results for 24-hour forecast windows.

## Research targets

The current regression experiment defines ten target conditions from
science-quality, one-minute GOES XRS-B observations. For forecast start `s`,
the prediction window is `[s, s + 24 h)`. The ten conditions belong to four
target families:

- **Maximum X-ray flux (`max_xflux`)**: maximum XRS-B irradiance in the
  prediction window.
- **Fluence (`fluence`)**: unscaled sum of valid one-minute XRS-B samples.
- **Maximum background deviation (`max_background_deviation`)**: maximum
  difference between XRS-B irradiance and the selected background level.
- **Total background deviation (`total_background_deviation`)**: unscaled
  one-minute sum of those differences.

The two deviation families each have four conditions, using background windows
of 2 or 24 hours and the mean of the lower 10% or 50% of valid samples. Each
background is calculated immediately before `s`. All available valid samples
are used even when coverage is below 90%; values are missing only when a
required window has no valid samples. For data preparation details and
reproduction instructions, see the [data preparation guide](scripts/data/README.md).

The v1 source samples follow the published temporal splits of the
[NASA-IBM SuryaBench flare-forecasting
dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting):
train, validation, leaky validation, and test. Target generation preserves
those source memberships.

Training uses the historical rows outside the seasonal holdout windows.
Validation contains January 15–31 rows for 2011–2019. The separate
`leaky_validation` split contains January 1–14 and February 1–14 rows for
2011–2019. Test contains all rows from 2020–2024. The leaky-validation split
is not used in the v1 model runs.

| Split | Time period and policy | CSV rows | Share of rows |
| --- | --- | ---: | ---: |
| Train | 2010-05-13–2019-12-31; historical training rows outside holdout windows | 74,760 | 58.26% |
| Validation | Jan 15–31 each year, 2011–2019 | 3,672 | 2.86% |
| Leaky validation | Jan 1–14 and Feb 1–14 each year, 2011–2019 | 6,048 | 4.71% |
| Test | 2020-01-01–2024-12-31; all timestamps | 43,848 | 34.17% |

Shares use 128,328 total split CSV rows. These counts precede image matching
and target-specific missing-value filtering. The final test summary reports
usable sample counts separately for each target and seed.

The v1 CompactCNN results use the training split for normalization and
validation for checkpoint selection. The final test metrics for all ten
targets and three seeds are in the
[test metrics summary](outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv).

## Data sources and methods

The project uses NOAA/NCEI [GOES XRS Level 2 one-minute measurements](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) to construct all ten target conditions.

Detailed formulas, quality filters, source versions, split handling, and
reproduction commands are documented separately:

- [Data-generation scripts and commands](scripts/data/README.md)

## Repository layout

```text
rxfi-benchmark/
├── configs/          # Data-generation configurations and input locks
├── data/             # Local source caches and derived targets (gitignored)
├── outputs/          # Reproducibility summaries and data-quality artifacts
├── scripts/          # Data download, background, and target-generation tools
└── environment.yml   # Conda environment for data workflows
```

## License

This code is released under the [Apache License 2.0](LICENSE). Users are
responsible for complying with the terms of external data providers.
