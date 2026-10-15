# rxfi-benchmark

This repository supports research on how the definition of a solar-flare
forecasting target affects its learnability and operational relevance. It
focuses on reproducible target definitions and data preparation for 24-hour
forecast windows; it does not present model architectures or model results.

## Research targets

The primary targets use science-quality, one-minute GOES XRS-B observations.
For a forecast start `s`, they use the future interval `[s, s + 24 h)` and a
fixed background from `[s - 24 h, s)`:

- `max_xflux`: maximum XRS-B irradiance.
- `fluence`: cadence-weighted sum of XRS-B irradiance.
- `max_background_deviation`: maximum signed difference from the background.
- `total_background_deviation`: cadence-weighted sum of signed differences.

A supplementary `max_old_rxfi` target supports comparison with the historical
catalog-event ratio `peak irradiance / background irradiance`. Its background
uses the 24 hours before flare start, excludes other flare intervals, and is
capped at the XRS-B value at the event's start. A split window includes events
whose catalog `start_time` falls in `[s, s + 24 h)`. Windows with no eligible
event RXFI are missing rather than zero. This supplemental target does not
change the four XRS-only definitions.

The samples follow the published temporal splits of the
[NASA-IBM SuryaBench flare-forecasting dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting): train, validation, leaky validation, and test. Split membership is preserved during target generation.

## Data sources and methods

The project uses NOAA/NCEI [GOES XRS Level 2 one-minute measurements](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) and the NOAA/NCEI [GOES XRS flare report](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv). The four primary targets are constructed from XRS measurements; the additional old-RXFI label is constructed from catalog events enriched with the historical background calculation.

Detailed formulas, quality filters, source versions, split handling, and
reproduction commands are documented separately:

- [XRS-only target definitions](docs/research/xrs_only_target_definitions.md)
- [Supplementary old-RXFI split target](docs/research/old_rxfi_split_targets.md)
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
