# rxfi-benchmark

> A benchmark for 24-hour solar-flare forecasting with Relative X-ray Flux Increase (RXFI).

`rxfi-benchmark` evaluates NOAA/NCEI-derived absolute and background-relative flare targets from frozen SuryaBench SDO image splits. It is a research repository for controlled target-representation and input-channel comparisons, not a production forecasting service.

## Overview

Forecasting the largest forthcoming GOES soft-X-ray peak captures absolute flare magnitude, but not its increase above the pre-flare X-ray background. RXFI adds that relative perspective. The benchmark holds image inputs, temporal splits, and model settings fixed while comparing what can be learned from SDO observations and how predicted continuous scores rank conventional severe-flare windows.

## RXFI

For an eligible NOAA flare-report event,

```math
\mathrm{RXFI} = \frac{F_{\mathrm{peak}} - F_{\mathrm{background}}}{F_{\mathrm{background}}}
```

$F_{\mathrm{peak}}$ is the one-minute GOES XRS-B peak irradiance (`xrsb_irrad`) and $F_{\mathrm{background}}$ is the selected background estimate, both in W m$^{-2}$. Project RXFI is the fractional increase above background: an RXFI of 4 means the peak is five times the background. Missing, non-finite, or non-positive inputs are invalid.

### Selected project background: lower-50% 24-hour mean

For flare start time $t_0$, the selected event-level background uses same-satellite GOES XRS-B `avg1m` observations in the preceding UTC interval $[t_0-24\,\mathrm{h},t_0)$. It retains finite positive measurements with `xrsb_flag == 0`, removes measurements within another catalog flare's inclusive start–end interval, then averages the lowest $\lceil N/2\rceil$ of the remaining $N$ measurements. This lower-50% mean is a project adaptation intended to reduce residual elevated flux after flare-interval masking. It does not apply a start-time upper cap and does not use `POST_EVENT`.

Project RXFI is the fractional excess $(F_{\mathrm{peak}}-B_{50})/B_{50}$; an RXFI of 4 means the peak is five times this background. The lower-50% procedure is distinct from the full arithmetic mean and optional start-time cap described by Hong et al., [*Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*](https://par.nsf.gov/servlets/purl/10435332), Section 2.1. NOAA's `background_irrad` remains a comparison field. Existing frozen forecast-window targets have not yet been regenerated: their RXFI values still use NOAA `background_irrad`. See the [selected lower-50% method and results](docs/research/lower50_background_rxfi.md), [paper procedure](docs/research/paper_based_background_xray_flux_method.md), and [paper capped/uncapped sensitivity analysis](docs/research/paper_background_cap_sensitivity.md).

The event-level lower-50% calculation is reproducible using the project environment and pinned NOAA inputs; its run configuration and outputs are in [`outputs/background_lower50/`](outputs/background_lower50/). It preserves a legacy calendar-year flare-interval lookup: near New Year's, an interval that began in the preceding year but overlaps the 24-hour window can be omitted. This limitation is recorded in the [method note](docs/research/lower50_background_rxfi.md).

The implemented 24-hour targets are `max_peak_flux`, `cumulative_peak_flux`, `max_rxfi`, and `cumulative_rxfi`. Maximum and cumulative forms distinguish the dominant eligible event from total activity in a forecast window.

## Benchmark at a glance

| Component | Implemented setting |
| --- | --- |
| Forecast window | Half-open future interval $[t, t + 24\ \mathrm{h})$ |
| Event source | NOAA/NCEI Level-2 GOES XRS Flare Report |
| Image input | SuryaBench 13-channel, 224$\times$224 SDO Zarr images |
| Model family | Randomly initialized `vit_small_patch16_224` |
| Classification target | NOAA-derived `max_flare_class`, five ordinal classes `FQ/A`, `B`, `C`, `M`, `X` |
| Regression targets | Maximum/cumulative peak flux and maximum/cumulative RXFI |
| Quantile regression | q05/q50/q95 with pinball loss |
| Split policy | Frozen temporal train, validation, test, and diagnostic leaky-validation splits |

## Benchmark pipeline

```mermaid
flowchart LR
    A[Frozen SuryaBench split timestamps] --> C[24-hour target generation]
    B[NOAA/NCEI GOES XRS flare report] --> C
    C --> D[Derived NOAA targets]
    E[SuryaBench SDO Zarr images] --> F[Config-driven ViT training]
    D --> F
    F --> G[Validation-selected checkpoints and prediction exports]
    G --> H[Regression and event-ranking analysis]
```

## Dataset and label construction

For each immutable split timestamp $t$, the generator aggregates eligible flare peaks in $[t,t+24\ \mathrm{h})$. `max_peak_flux` and `cumulative_peak_flux` use valid positive XRS-B peaks; `max_rxfi` and `cumulative_rxfi` use valid event-level RXFI values. The frozen target files currently use the historical NOAA `background_irrad` RXFI definition; the new lower-50% event-level results have not been propagated to those forecast-window labels. `max_flare_class` is the NOAA class of the maximum-peak event; a no-event window is `FQ`. Ties are resolved by earliest peak time, then stable event identifier.

The corrected ordinal classifier uses this explicit fixed mapping:

| Band | Index |
| --- | ---: |
| `FQ` or `A*` | 0 (`FQ/A`) |
| `B*` | 1 |
| `C*` | 2 |
| `M*` | 3 |
| `X*` | 4 |

The raw NOAA class remains in `max_flare_class`; classifier label encoding merges `FQ` and all A-class magnitudes into the shared `FQ/A` category. Values such as `B3.4` and `M2.1` map to their broad bands, and malformed labels are rejected. New five-class experiments use the `_ordinal5` configurations; `scripts/launchers/project2/run_project2_preliminary.sh classification` and the Slurm preliminary array select these configurations. Previously completed `_ordinal6` results and their configs are retained as historical six-class experiments and are not rewritten by this mapping change.

See the [target-generation specification](docs/research/24h_target_generation.md) and [NOAA XRS/RXFI definition](docs/research/noaa_xrs_catalog_and_rxfi_definition.md) for quality rules, no-flare handling, and catalog caveats.

## Implemented experimental settings

| Family | Target(s) | Controlled comparison |
| --- | --- | --- |
| Project 1 QR | Four peak-flux/RXFI targets | Same 13-channel input, ViT-Small/16, resolution, quantiles, optimizer, seed, and validation-pinball selection rule |
| Project 2 classification | Ordinal `max_flare_class` | `hmi_m`, `aia131`, `aia193`, and `all13`; validation macro-F1 selection |
| Project 2 QR | `cumulative_peak_flux` | The same four input conditions; validation pinball-loss selection |

Project 1 transforms are `log10(max_peak_flux / 1e-8)`, `log10(1 + cumulative_peak_flux / 1e-8)`, `log10(1 + max_rxfi)`, and `log10(1 + cumulative_rxfi)`.

## Repository structure

```text
rxfi-benchmark/
├── configs/          # YAML experiment configurations
├── data/             # local derived targets and raw caches (gitignored)
├── docs/research/    # specifications and research notes
├── scripts/          # topic-organized scripts; see scripts/README.md for the directory map
├── tests/            # focused implementation checks
├── environment.yml   # Conda environment
└── LICENSE           # Apache-2.0 license
```

## Installation

The checked-in Conda environment specifies Python 3.11.16, PyTorch 2.5.1 CUDA 12.4 wheels, `timm`, `zarr`, and `scikit-learn`.

```bash
git clone git@github.com:JinsuHongg/rxfi-benchmark.git
cd rxfi-benchmark
conda env create -f environment.yml
conda activate rxfi-benchmark
```

The repository does not distribute the SuryaBench Zarr store, frozen split CSVs, derived target CSVs, or NOAA catalog files. Point the selected YAML configuration at authorized local copies.

## Quick start

### Reproduce the paper-based background calculation

After activating the project environment, run this from the repository root:

```bash
conda activate rxfi-benchmark
python scripts/background/compute_paper_background_sensitivity.py --offline
```

Offline mode requires the pinned NOAA flare catalog and all required GOES XRS Level 2 files to be present in `data/raw/noaa_paper_background/`. Their catalog release, XRS product version, sizes, and SHA-256 checksums are pinned in [`configs/paper_background_sensitivity.json`](configs/paper_background_sensitivity.json) and its [input lock](configs/paper_background_sensitivity.inputs.lock.json). To allow the script to download missing locked inputs, omit `--offline`.

Each run writes `event_backgrounds.csv`, `summary.json`, `input_manifest.json`, and `run_config.json` under `outputs/paper_background_sensitivity/runs/<fingerprint>/`. The fingerprint identifies the configuration and exact inputs; the frozen root-level reference files are left unchanged. This command calculates event-level backgrounds only. It does not aggregate forecast windows or regenerate derived targets. See the [full run guide](outputs/paper_background_sensitivity/README.md) for custom paths, checksum behavior, and overwrite options.

### Generate forecast-window targets

Generate targets from immutable splits and a local NOAA flare-report CSV:

```bash
python scripts/data/generate_24h_targets.py \
  --data-dir /path/to/index_data \
  --noaa-csv /path/to/noaa_flare_report.csv \
  --derived-dir data/derived
```

Validate transforms without training:

```bash
python scripts/training/train_vit_quantile_regression.py \
  --config configs/vit_small_224_max_peak_flux_qr.yaml \
  --validate-transforms
```

After the YAML paths resolve and a CUDA GPU is available, run a representative QR experiment:

```bash
python scripts/training/train_vit_quantile_regression.py \
  --config configs/vit_small_224_cumulative_peak_flux_qr.yaml
```

The scripts require CUDA. Cluster launchers are at [scripts/launchers/run_all_qr_experiments.sh](scripts/launchers/run_all_qr_experiments.sh), [scripts/launchers/project2/run_project2_preliminary.sh](scripts/launchers/project2/run_project2_preliminary.sh), and [scripts/launchers/project2/slurm/](scripts/launchers/project2/slurm/).

## Configuration and evaluation

YAML files define data paths, target transform or representation, selected channels, model, image size, seed, and output directory. Training supports `--batch-size`, `--precision` (`32`, `16-mixed`, `bf16-mixed`), `--num-workers`, and `--gradient-accumulation-steps` overrides.

Classification reports accuracy, balanced accuracy, macro-F1, class-wise precision/recall/F1, and a test confusion matrix. Quantile regression reports per-quantile pinball loss, q50 errors, empirical quantile frequencies, and crossing rates. The Project 1 analysis additionally uses transformed-space Spearman correlation, $R^2$, MAE/IQR, and q50-based M+/X+ AUROC/AUPRC.

## Current benchmark status

> [!NOTE]
> This repository is under active research development. Benchmark definitions, preprocessing, and experimental settings may evolve.

- **Implemented:** NOAA/XRS target construction, reproducible lower-50% event-background/RXFI calculation, ViT-Small classification and QR pipelines, and Project 2 channel-comparison configurations. The frozen forecast-window RXFI targets still use the historical NOAA background and await regeneration.
- **Documented results:** the controlled Project 1 four-target QR analysis is available in the [predictability and event-relevance report](docs/research/four_qr_target_predictability_and_event_relevance.md).
- **Classification caution:** original Project 2 outputs used exact NOAA magnitude strings; the later six-band runs are historical. New runs use the five-class `FQ/A`, `B`, `C`, `M`, `X` mapping.
- **TBD:** a fixed OCQR calibration/evaluation procedure is not implemented.

## Results

The documented Project 1 QR comparison uses the same 27,980-row test cohort for all four targets. Its RXFI targets use the historical NOAA `background_irrad` baseline; they have not been retrained with the selected lower-50% background. The analysis separates learnability from severe-event relevance and does not establish a globally optimal physical target.

| Target | Spearman | MAE / IQR | M+ AUPRC | X+ AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Maximum peak flux | 0.826 | 0.244 | 0.655 | 0.093 |
| Cumulative peak flux | 0.859 | 0.247 | 0.715 | 0.148 |
| Maximum RXFI | 0.509 | 0.481 | 0.613 | 0.108 |
| Cumulative RXFI | 0.576 | 0.402 | 0.581 | 0.095 |

See the [four-target QR report](docs/research/four_qr_target_predictability_and_event_relevance.md) for $R^2$, no-skill AUPRC baselines, quantile diagnostics, and limitations.

## Reproducibility

Training records resolved configurations, source-split hashes, row-alignment checks, selected channels, cohort availability, seed, GPU/runtime metadata, learning logs, checkpoint metadata, and prediction exports. Frozen split hashes are checked before training; test data are not used for checkpoint selection. Temporal split rules are documented in the [RXFI distribution and imbalance analysis](docs/research/rxfi_distribution_and_imbalance_analysis.md).

## Documentation

- [Selected lower-50% background method and event-level results](docs/research/lower50_background_rxfi.md)
- [Paper-based 24-hour background method and source](docs/research/paper_based_background_xray_flux_method.md)
- [Paper-based capped/uncapped background comparison](docs/research/paper_background_cap_sensitivity.md)
- [Lower-50% calculation run guide and results](outputs/background_lower50/README.md)
- [Background calculation run guide](outputs/paper_background_sensitivity/README.md)
- [NOAA XRS catalog and RXFI definition](docs/research/noaa_xrs_catalog_and_rxfi_definition.md)
- [Deterministic 24-hour target generation](docs/research/24h_target_generation.md)
- [Four-target ViT-Small quantile regression](docs/research/vit_small_224_four_target_quantile_regression.md)
- [Project 2 preliminary multichannel experiment](docs/research/project2_preliminary_multichannel_experiment.md)
- [Current research state](docs/research/current_state.md)

## Citation

Citation information will be added when the associated paper becomes available.

## License

This code is released under the [Apache License 2.0](LICENSE). Users are responsible for complying with the terms of external data providers.

## Acknowledgments

Flare targets use the NOAA/NCEI GOES XRS Flare Report. See the [catalog specification](docs/research/noaa_xrs_catalog_and_rxfi_definition.md) for provenance and caveats.
