# rxfi-benchmark

This repository supports research on how the definition of a solar-flare
forecasting target affects its learnability and operational relevance. It
focuses on reproducible target definitions and data preparation for 24-hour
forecast windows; it does not present model architectures or model results.

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
required window has no valid samples. Full formulas, coverage definitions,
quality rules, source versions, and reproduction instructions are in the
[XRS-only target definitions](docs/research/xrs_only_target_definitions.md).

The source samples follow the published temporal splits of the
[NASA-IBM SuryaBench flare-forecasting
dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting):
train, validation, leaky validation, and test. Target generation preserves
those source memberships.

The current derived temporal split assigns historical rows through 2019 to
train, validation, or calibration (`leaky_validation`). Validation
uses January 15–February 14 and July 15–August 14 each year. Calibration uses
the ten days immediately before and after those windows. Only calibration rows
whose 24-hour target support overlaps a usable train or validation sample
are removed. Train and validation rows are retained, and the existing
test CSV is preserved byte-for-byte.

| Split | CSV rows | Share of rows | Usable target and image samples* |
| --- | ---: | ---: | ---: |
| Train | 61,224 | 48.36% | 58,007 |
| Validation | 14,136 | 11.17% | 13,324 |
| Leaky validation | 7,382 | 5.83% | 6,942 |
| Test | 43,848 | 34.64% | 41,910 |

Shares use 126,590 total CSV rows. *Usable samples have a finite `max_xflux`
target and a matching image timestamp in the Zarr store. The train/test
boundary retains 23 train and 21 test samples whose 24-hour target supports
intersect (270 interval pairs). The detailed split policy, audit, and
reproduction commands are in the
[data preparation README](scripts/data/README.md#seasonal-validation-split-v2).

## Data sources and methods

The project uses NOAA/NCEI [GOES XRS Level 2 one-minute measurements](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) and the NOAA/NCEI [GOES XRS flare report](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv). The ten primary target conditions are constructed from XRS measurements; the additional old-RXFI label is constructed from catalog events enriched with the historical background calculation.

Detailed formulas, quality filters, source versions, split handling, and
reproduction commands are documented separately:

- [XRS-only target definitions](docs/research/xrs_only_target_definitions.md)
- [Supplementary old-RXFI split target](docs/research/old_rxfi_split_targets.md)
- [Legacy RXFI versus background-deviation comparison](docs/research/old_rxfi_vs_background_deviation.md)
- [NOAA catalog fields and historical RXFI context](docs/research/noaa_xrs_catalog_and_rxfi_definition.md)
- [Background calculation and event-level catalog enrichment](scripts/background/README.md)
- [Data-generation scripts and commands](scripts/data/README.md)
- [Legacy catalog-based target generation](docs/research/24h_target_generation.md)

## Repository layout

```text
rxfi-benchmark/
├── configs/          # Data-generation configurations and input locks
├── data/             # Local source caches and derived targets (gitignored)
├── docs/research/    # Research definitions, provenance, and methods
├── outputs/          # Reproducibility summaries and data-quality artifacts
├── scripts/          # Data download, background, and target-generation tools
└── environment.yml   # Conda environment for data workflows
```

## License

This code is released under the [Apache License 2.0](LICENSE). Users are
responsible for complying with the terms of external data providers.
