# rxfi-benchmark

This repository contains reproducible data definitions and target-generation code for 24-hour solar-flare forecasting windows. It combines frozen temporal splits from the [NASA-IBM SuryaBench flare-forecasting dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting) with NOAA/NCEI GOES X-ray flare reports and GOES XRS Level 2 measurements.

## Data sources and temporal splits

The forecasting samples use the published SuryaBench split CSVs. The split membership follows the source dataset's temporal policy:

| Split | Time range and rule | Rows |
| --- | --- | ---: |
| Train | February 15–December 31, 2010–2019 | 74,760 |
| Validation | January 15–31, 2010–2019 | 3,672 |
| Leaky validation | January 1–14 and February 1–14, 2010–2019 | 6,048 |
| Test | January 1, 2020–December 31, 2024 | 43,848 |

The four split files contain 128,328 hourly timestamps in total. They are read as immutable inputs; generated files preserve each input row, column, timestamp, and row order. Their source and SHA-256 checksums are recorded in [`target_generation_metadata.json`](outputs/target_generation_24h/target_generation_metadata.json). The source split files and generated CSVs are local data products and are not included in Git.

Flare events and peak irradiance come from the NOAA/NCEI GOES XRS Flare Report. Event backgrounds use science-quality GOES XRS Level 2 `avg1m` XRS-B observations (`xrsb_flux`, `xrsb_flag`, and the catalog-selected satellite). See the [catalog and XRS field definition](docs/research/noaa_xrs_catalog_and_rxfi_definition.md) and the input lock at [`configs/paper_background_sensitivity.inputs.lock.json`](configs/paper_background_sensitivity.inputs.lock.json).

## Forecast-window targets

For every split timestamp `t`, events with catalog peak times in the half-open interval `[t, t + 24 h)` contribute to the four continuous targets:

| Target | Definition |
| --- | --- |
| `max_peak_flux` | Largest valid positive NOAA event peak in the window, in W m^-2 |
| `cumulative_peak_flux` | Sum of valid positive event peaks in the window, in W m^-2 |
| `max_rxfi` | Largest event RXFI in the window |
| `cumulative_rxfi` | Sum of event RXFIs in the window |

The background-relative targets now use the project's selected lower-50% background. For each flare, event background `B50` is calculated from same-satellite, positive finite XRS-B `avg1m` samples in the 24 hours before flare start. Samples with `xrsb_flag != 0` and samples inside another catalog flare's start–end interval are removed. The arithmetic mean of the lowest `ceil(N/2)` remaining measurements is `B50`. RXFI is the fractional increase above that background:

\[
\mathrm{RXFI}_{50}=\frac{F_{\mathrm{peak}}-B_{50}}{B_{50}}.
\]

This lower-50% estimator is a project adaptation intended to reduce residual flare elevation after event-interval masking. It does not use NOAA `background_irrad`, does not apply the paper's start-time upper cap, and does not use `POST_EVENT`. The published paper's filtered 24-hour background procedure is summarized in [the method note](docs/research/paper_based_background_xray_flux_method.md); the selected project adaptation and its caveats are documented in [lower50_background_rxfi.md](docs/research/lower50_background_rxfi.md).

No-event windows have zero for all four continuous targets and `FQ` for `max_flare_class`. **Negative event-level RXFI values do occur:** 261 of 41,723 catalog events in the 2010–2025 source period (0.63%) have a peak below `B50`; their calculated values remain in the event-level `event_backgrounds.csv`. For the split targets, these events are excluded from `max_rxfi` and `cumulative_rxfi`, but remain in the peak-flux targets. If a window has no non-negative RXFI events, both RXFI targets are zero, so generated RXFI targets are never negative.

The current split-specific files are `data/derived/{train,validation,test,leaky_validation}_targets.csv`. Target summaries, row-alignment checks, quality counts, and figures are in [`outputs/target_generation_24h/`](outputs/target_generation_24h/). Local target CSVs are ignored by Git because they are reproducible data products.

## Reproduce the data products

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate rxfi-benchmark
```

Download the four source split files from the dataset repository into the local raw-data directory:

```bash
mkdir -p data/raw/surya_bench_splits
for split in train validation test leaky_validation; do
  curl -L "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting/resolve/main/${split}.csv" \
    -o "data/raw/surya_bench_splits/${split}.csv"
done
```

Recalculate the lower-50% event backgrounds using the pinned NOAA flare report and XRS inputs. Offline mode requires those inputs in `data/raw/noaa_paper_background/`; omit `--offline` to download missing pinned inputs:

```bash
python scripts/background/compute_paper_background_sensitivity.py \
  --config configs/background_lower50.json --offline
```

The command writes `event_backgrounds.csv` under `outputs/background_lower50/runs/<fingerprint>/`. Use that file and the matching NOAA catalog to regenerate the four split targets:

```bash
python scripts/data/generate_24h_targets.py \
  --data-dir data/raw/surya_bench_splits \
  --noaa-csv data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv \
  --rxfi-method lower50 \
  --lower50-backgrounds-csv outputs/background_lower50/runs/<fingerprint>/event_backgrounds.csv \
  --derived-dir data/derived \
  --output-dir outputs/target_generation_24h
```

Replace `<fingerprint>` with the run directory created by the background calculation. The generator records checksums for the input catalog, background table, and split files; it also checks that split row identity is preserved. See the [target-generation specification](docs/research/24h_target_generation.md) for details.

## Repository layout

```text
rxfi-benchmark/
├── configs/          # Pinned data and background configurations
├── data/             # Local source caches and derived targets (gitignored)
├── docs/research/    # Data definitions, provenance, and methodology
├── outputs/          # Reproducibility summaries and data-quality artifacts
├── scripts/          # Data download, background, and target-generation tools
└── environment.yml   # Conda environment for data workflows
```

## Data documentation

- [NOAA XRS catalog fields and RXFI definition](docs/research/noaa_xrs_catalog_and_rxfi_definition.md)
- [Selected lower-50% background method and event results](docs/research/lower50_background_rxfi.md)
- [Paper-based 24-hour background procedure](docs/research/paper_based_background_xray_flux_method.md)
- [Deterministic 24-hour target generation](docs/research/24h_target_generation.md)
- [Background calculation run guide](outputs/background_lower50/README.md)

## License

This code is released under the [Apache License 2.0](LICENSE). Users are responsible for complying with the terms of external data providers.
