# Scripts Topic Organization Design

## Goal

Reorganize the repository's scripts into topic-based directories, give every directory in the reorganized `scripts/` tree a README, and update all references so documented commands, tests, and launchers continue to work.

## Approved direction

The user approved organizing by topic and updating references to the new paths. No compatibility wrappers will remain at the old root paths; the new paths become canonical. This avoids a root directory containing moved-script stubs, at the cost of requiring callers to update their commands.

## Proposed layout and file ownership

| Destination | Files |
| --- | --- |
| `scripts/background/` | `compute_paper_background_sensitivity.py`; `paper_background/` package (`__init__.py`, `calculation.py`, `noaa.py`, `runner.py`) |
| `scripts/data/` | `generate_24h_targets.py`, `inspect_noaa_xrs_flare_report.py`, `refine_no_flare_max_peak_flux.py` |
| `scripts/analysis/` | `analyze_four_qr_targets.py`, `analyze_rxfi_distribution.py`, `audit_fq_xrs_threshold_exceedance.py`, `audit_historical_goes_label_mismatch.py`, `evaluate_severe_event_binary_metrics.py`, `severe_event_metrics.py` |
| `scripts/analysis/project2/` | `analyze_project2_preliminary.py`, `audit_project2_channels.py` |
| `scripts/visualization/` | `plot_four_24h_targets_noaa.py`, `plot_four_24h_targets_xrs_1min.py`, `plot_project1_severe_event_metrics.py`, `plot_single_flare_classification.py` |
| `scripts/training/` | `benchmark_qr_batch_size.py`, `train_vit_classifier.py`, `train_vit_quantile_regression.py` |
| `scripts/launchers/` | `run_all_qr_experiments.sh`, `slurm_vit_small_224_max_flare_class.sbatch` |
| `scripts/launchers/slurm/` | `run_qr_array.sbatch` |
| `scripts/launchers/project2/` | `run_project2_preliminary.sh` |
| `scripts/launchers/project2/slurm/` | `run_project2_ordinal6_classification_array.sbatch`, `run_project2_preliminary_array.sbatch` |
| `scripts/exports/project3/` | `export_project3_predictions.py`, `project3_export_contract.py`, `project3_export_loader_example.py`, `validate_project3_exports.py` |

Add `scripts/README.md` as a directory index and add a README in every listed directory, including nested package/launcher directories. Each README states the directory's purpose and briefly describes the role of every file in that directory. Include repository-root invocation examples where applicable.

## Reference and runtime updates

- Update root README commands and links, research-document links/commands, tests' imports and `sys.path` setup, and shell/Slurm commands to the new paths.
- Preserve repository-root behavior for data/config/output paths. Update script `ROOT` calculations after relocation, including the deeper `analysis/project2/` and `exports/project3/` modules.
- Preserve direct execution of Python CLIs with `python scripts/<topic>/<script>.py`; keep colocated imports working for the background package and severe-event analysis helpers.
- Do not rename configs, data, output artifacts, model code, or script behavior as part of this organization change.

## Verification

- Search the repository for references to every old script path and resolve remaining references intentionally (historical plan prose may remain as historical text; current usage docs and executable launchers must point to new locations).
- Run focused tests for NOAA/background, Project 2 classifiers/analysis, severe-event metrics, Project 3 export contract, and plot helpers.
- Run `--help` for relocated CLIs that support it, shell syntax checks for launchers, and `git diff --check`.
- Run the full available pytest suite and record any pre-existing environment failures separately from failures caused by relocation.

## Out of scope

- Compatibility aliases at old script paths.
- Behavioral changes to data generation, background calculation, analysis, visualization, training, or export logic.
- Changes to configurations, datasets, model weights, or generated results beyond updating documentation to point at moved scripts.
