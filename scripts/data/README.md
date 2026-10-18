# Data preparation

Scripts for creating and inspecting regression targets from the frozen temporal
splits and NOAA GOES XRS measurements.

- `generate_xrs_only_targets.py`: reproducibly downloads the four SuryaBench split indexes and NOAA/NCEI science-quality GOES XRS Level 2 one-minute XRS-B files, then adds `max_xflux`, `fluence`, `max_background_deviation`, and `total_background_deviation`. It uses `[s,s+24h)` for prediction and the lower-50% mean from the immediately preceding interval `[s-24h,s)` as background. At each minute it uses a flag-zero sample from the scheduled primary, falling back to the scheduled secondary whenever the primary has no flag-zero sample. If neither satellite has flag-zero data, the minute is skipped and recorded in `missing_xrs_samples.csv`; generation stops before replacing split files only when a required target window has no valid samples. Per-window coverage and source counts record primary and fallback use. Existing catalog-based `*_targets.csv` files are backed up once to `data/derived/legacy_catalog_targets/` before replacement. Run metadata records exact source URLs and SHA-256 hashes.
- `generate_regression_targets.py`: generates the ten approved experiment target conditions from the four frozen SuryaBench splits and science-quality one-minute XRS-B. The two base targets are `max_xflux` and unscaled sample-sum `fluence`; the two signed background-deviation targets are generated for 2h/24h windows and lower 10%/50% means. It writes the concise split names `train.csv`, `validation.csv`, `leaky_validation.csv`, and `test.csv` directly under `data/derived/`, without changing `*_targets.csv`. Per-window coverage/QC, source checksums, and config provenance go to `outputs/regression_target_experiment/runs/<run_id>/`.
- `add_old_max_rxfi_targets.py`: reads the enriched NOAA catalog copy produced by `scripts/background/add_old_background_rxfi_to_catalog.py` and adds `max_old_rxfi` to separate copies of the four XRS-only target splits. Window membership uses catalog `start_time` in `[t,t+24h)`. Windows without a catalog event or without valid old RXFI remain missing, not zero. Per-split counts, source hashes, and the aggregation rule are recorded.
- `generate_24h_targets.py` (**legacy**): builds the catalog-based peak-flux/RXFI targets from frozen splits, the NOAA flare report, and checksum-pinned XRS-B measurements. For FQ windows it finds the in-window XRS maximum and calculates lower-50% background over the 24 hours ending five minutes before that maximum.
- `inspect_noaa_xrs_flare_report.py`: inspects flare-report fields and historical NOAA `background_irrad`/RXFI values. The selected event-background calculation is in [`../background/README.md`](../background/README.md).
- `refine_no_flare_max_peak_flux.py`: fills no-flare-window maximum flux from quality-filtered GOES XRS one-minute data and writes coverage/QC outputs.

Run the new generator from the repository root with the project environment:

```bash
conda run -n rxfi-benchmark python scripts/data/generate_xrs_only_targets.py
```

Generate the ten regression experiment targets separately, preserving the
existing `*_targets.csv` files:

```bash
conda run -n rxfi-benchmark python -m scripts.data.generate_regression_targets
```

On the cluster, submit the CPU launcher instead:

```bash
mkdir -p logs/target_generation
sbatch scripts/launchers/slurm/generate_regression_targets.sbatch
```

The launcher uses the `flare_surya` environment unless
`CONDA_ENV_NAME` names another installed environment. It does not pass
`--overwrite`; existing split files cause the generator to stop.

The experiment settings are in
[`../../configs/regression_target_experiment.json`](../../configs/regression_target_experiment.json).
Use `--revision <40-character-commit-sha> --offline` to reproduce using cached
inputs only. The script refuses to replace existing `data/derived/{train,
validation,leaky_validation,test}.csv` unless `--overwrite` is provided. Its
run metadata records the resolved Hugging Face revision, NOAA input hashes,
split output hashes, target formulas, and per-window coverage.

After generating the event-level old background and RXFI catalog copy, add the
supplementary maximum old-RXFI target without replacing the XRS-only files:

```bash
conda run -n rxfi-benchmark python scripts/data/add_old_max_rxfi_targets.py
```

The default inputs are `data/derived/noaa_flare_catalog_old_background_rxfi.csv`
and `data/derived/{train,validation,leaky_validation,test}_targets.csv`. The
augmented copies, quality-control table, and metadata are written to
`data/derived/old_rxfi_targets/`. Use `--overwrite` to regenerate existing
outputs. The event-level method and label semantics are specified in
[`../../docs/research/old_rxfi_split_targets.md`](../../docs/research/old_rxfi_split_targets.md).

Use `--revision <40-character-commit-sha> --offline` to reproduce from cached
inputs only. NOAA's direct annual file layout is generation-specific:

- GOES-8–15: [NCEI legacy XRS archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/), using `v2-2-1` annual files.
- GOES-16–18: [NOAA/NCEI GOES-R archive](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/), using `v2-2-2` annual files.
- GOES-19: the same GOES-R archive, using `v2-2-1` annual files.
- The generator selects the version per satellite and records it in run metadata because NOAA may publish versions on different schedules.
- Minutes with no flag-zero sample from either scheduled satellite are listed with timestamp and per-satellite status in `missing_xrs_samples.csv`. All split rows are retained; target values are blank only for targets whose required prediction or background window has no valid samples.

The primary/secondary schedule comes from [NOAA's GOES Flare Report ReadMe,
Appendix A, Table 5](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf).
Run metadata contains each exact annual file URL and SHA-256. See the
[XRS-only target definition](../../docs/research/xrs_only_target_definitions.md)
for formulas, sources, and QC fields. Legacy generation is described in the
[legacy target-generation notes](../../docs/research/24h_target_generation.md).
