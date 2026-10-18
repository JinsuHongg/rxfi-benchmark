# XRS-only regression target experiment

## Scope and status

This document defines the ten target conditions in the current regression
target experiment. Targets are calculated from science-quality, one-minute
GOES XRS Level 2 XRS-B observations. The NOAA flare catalog is not used to
construct these targets. It may be used separately to define flare outcomes
for operational-skill evaluation.

The reproducible generator is
[`scripts/data/generate_regression_targets.py`](../../scripts/data/generate_regression_targets.py),
and its frozen settings are in
[`configs/regression_target_experiment.json`](../../configs/regression_target_experiment.json).
The script writes `train.csv`, `validation.csv`, `leaky_validation.csv`, and
`test.csv` to `data/derived/`. Per-window coverage and source/output
provenance are recorded in
`outputs/regression_target_experiment/runs/<run_id>/`.

## Prediction window, background, and valid samples

For a forecast start time `s`, the prediction window is the half-open interval
`[s, s + 24 h)`. An eligible sample has finite, positive XRS-B flux and
`xrsb_flag == 0`. Scheduled primary GOES observations are preferred; a valid
secondary-satellite observation is used when the primary sample is missing or
fails the quality rule.

Each background is fixed over the prediction window and is calculated from
the immediately preceding interval `[s - W, s)`, where `W` is 2 or 24 hours.
If the background window has at least one eligible sample, let `N` be its
sample count. For lower-tail percentage `q` equal to 10 or 50, sort those
samples as `R_(1) <= ... <= R_(N)` and average the lowest
`k = max(1, ceil(q N / 100))` samples. With no eligible background samples,
the background is undefined:

\[
B_{W,q}(s) = \frac{1}{k} \sum_{j=1}^{k} R_{(j)}.
\]

The four background conditions are `2h_q10`, `2h_q50`, `24h_q10`, and
`24h_q50`. These intervals are anchored to the forecast start `s`, not to a
flare start. No flare intervals are masked.

### Coverage and missing observations

Coverage is the fraction of expected one-minute bins with eligible XRS-B
samples. The prediction-window coverage is `n_prediction / 1440`; background
coverage is `n_background / 120` for 2 hours and `n_background / 1440` for 24
hours. A coverage value of 0.9 therefore means 90% of the expected minute
samples are valid.

Only finite, positive measurements with quality flag 0 are used. If a scheduled
primary record has a measured flux but a nonzero flag, a flag-zero sample from
the scheduled secondary is used when available. If the primary timestamp or
flux value is missing, or neither satellite has a valid flag-zero sample, the
generator reports the mismatch and aborts before writing split files. Missing
samples are not interpolated or silently omitted. The QC file records valid
sample counts, coverage fractions, and primary/secondary source counts.

## Target conditions

The two background-independent targets are:

| Target | Definition | Units / scale |
| --- | --- | --- |
| `max_xflux` | `max_i F_i` | W m^-2 |
| `fluence` | `sum_i F_i` | Unscaled one-minute sample sum |

For each of the four background conditions, the experiment adds two signed
deviation targets:

| Target family | Definition for each `W,q` condition | Units / scale |
| --- | --- | --- |
| `max_background_deviation` | `max_i (F_i - B_(W,q)(s))` | W m^-2 |
| `total_background_deviation` | `sum_i (F_i - B_(W,q)(s))` | Unscaled one-minute sample sum |

Here, `F_i` are eligible samples in `[s, s + 24 h)`. The deviation targets
are emitted with the condition suffix, for example
`max_background_deviation_2h_q10` and
`total_background_deviation_24h_q50`. Together with `max_xflux` and `fluence`,
these make ten target columns.

The sums are stored without multiplying by 60 seconds. `fluence` and
`total_background_deviation` are therefore unscaled sample sums, not SI time
integrals in J m^-2. This scale convention preserves the numerical target
defined for the experiment. Negative deviations and negative total deviations
are retained; they are not clipped to zero.

## Data sources and satellite selection

The generator follows the primary/secondary GOES schedule in NOAA/NCEI's
[GOES Flare Report ReadMe, Appendix A, Table 5](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf).
For each minute, it uses a valid flag-zero sample from the scheduled primary
satellite. If the primary record exists with a measured flux but has a
nonzero quality flag, it uses a valid flag-zero sample from the scheduled
secondary when available. A missing primary timestamp or flux value causes an
error even if the secondary has data; having no valid flag-zero sample from
either satellite also stops generation. These checks run over the full required
time span before any split file is written. The flare-catalog satellite field
is not used to select the XRS series.

NOAA/NCEI's direct annual file layout depends on the GOES generation:

| Satellites | Official directory and direct-file pattern | Examples |
| --- | --- | --- |
| GOES-8–15 | [NCEI science/XRS archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/) — annual `v2-2-1` files | [GOES-15, 2012](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g15_y2012_v2-2-1.nc) (primary); [GOES-14, 2012](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes14/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g14_y2012_v2-2-1.nc) (secondary) |
| GOES-16 and GOES-18 | [NOAA/NCEI GOES-R archive](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/) — annual `v2-2-2` files | [GOES-16, 2023](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g16_y2023_v2-2-2.nc) (primary); [GOES-18, 2023](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g18_y2023_v2-2-2.nc) (secondary) |
| GOES-17 and GOES-19 | The same GOES-R archive — annual `v2-2-1` files | |

The annual product version is pinned per satellite: GOES-8–15, GOES-17, and
GOES-19 use `v2-2-1`; GOES-16 and GOES-18 use `v2-2-2`. Run metadata records
the product version for each NOAA file, the resolved Hugging Face revision,
source URLs and SHA-256 hashes, target file hashes, and the NOAA schedule
source.

## Reproducing target generation

From the repository root, use the project environment:

```bash
conda run -n rxfi-benchmark python -m scripts.data.generate_regression_targets
```

The generator reads the frozen SuryaBench temporal split indexes, downloads
missing NOAA inputs, and uses the settings in
[`regression_target_experiment.json`](../../configs/regression_target_experiment.json).
For a cache-only reproduction, provide a full Hugging Face commit SHA and add
`--offline`:

```bash
conda run -n rxfi-benchmark python -m scripts.data.generate_regression_targets \
  --revision <40-character-commit-sha> --offline
```

The script refuses to overwrite existing split CSVs unless `--overwrite` is
provided. See [`scripts/data/README.md`](../../scripts/data/README.md) for
details.

## Relationship to catalog-based RXFI

The historical `max_old_rxfi` target is generated separately from NOAA flare
catalog events and does not change the ten XRS-only target conditions. Its
construction is documented in
[`old_rxfi_split_targets.md`](old_rxfi_split_targets.md). The comparison of
that label with `max_background_deviation` is reported in
[`old_rxfi_vs_background_deviation.md`](old_rxfi_vs_background_deviation.md).
