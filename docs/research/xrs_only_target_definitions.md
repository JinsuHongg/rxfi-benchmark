# XRS-only 24-hour regression targets

## Status and scope

This document records the approved research definition and reproducible
implementation for four continuous targets. Values are generated only from
science-quality, one-minute GOES XRS Level 2 XRS-B observations. The
supplementary `max_old_rxfi` label is generated separately from NOAA flare
catalog events; it does not change these four XRS-only target definitions.

The generator is [`scripts/data/generate_xrs_only_targets.py`](../../scripts/data/generate_xrs_only_targets.py).
It writes the four split CSVs to `data/derived/` and archives previous
catalog-based files in `data/derived/legacy_catalog_targets/` before replacing
them. Run metadata and per-window coverage are stored under
`outputs/xrs_only_target_generation/runs/<run_id>/`. Catalog-based labels and
the supplementary `max_old_rxfi` target are documented separately in
[24h_target_generation.md](24h_target_generation.md) and
[old_rxfi_split_targets.md](old_rxfi_split_targets.md).

## Windows, background, and quality filtering

For a forecast-window start `s`, the prediction window is the half-open
interval `[s, s + 24 h)`. Let `F_i` denote eligible one-minute XRS-B samples in
that interval. A measurement is eligible if its flux is finite and positive
and `xrsb_flag == 0`.

The background is fixed for the whole prediction window. It is the mean of the
lower 50% of eligible samples in the 24-hour interval immediately preceding
the forecast window, `[s - 24 h, s)`. If the sorted samples are
`R_(1) <= ... <= R_(N)`, then

\[
B_s = \frac{1}{\lceil N/2\rceil}\sum_{j=1}^{\lceil N/2\rceil} R_{(j)}.
\]

This interval is based on forecast start `s`, not a flare start time. No flare
intervals are masked because the catalog is not used to construct these
targets.

## Four targets

| Target | Definition | Units |
| --- | --- | --- |
| `max_xflux` | `max_i F_i` | W m^-2 |
| `fluence` | `60 s * sum_i F_i` | J m^-2 |
| `max_background_deviation` | `max_i (F_i - B_s)` | W m^-2 |
| `total_background_deviation` | `60 s * sum_i (F_i - B_s)` | J m^-2 |

The sums use eligible one-minute samples in `[s, s + 24 h)`. The 60-second
factor treats each one-minute average as one minute of exposure, making the
result a cadence-weighted discrete approximation to a time integral.

Missing samples are not interpolated. Targets use available valid observations,
and the QC CSV records sample counts and coverage fractions for both windows.
No minimum coverage threshold is applied, and partial-window sums are not
scaled to 24 hours. Targets that depend on an empty sample set are left
missing. Background deviation is signed: negative sample deviations and
negative window targets are retained and never clipped to zero.

`max_background_deviation` is the largest signed difference from the fixed
background, not a fractional increase. The distinct fractional-excess formula
`(F - B_s) / B_s` is not one of these four targets.

## Satellite selection and source links

The generator follows the primary/secondary GOES schedule in NOAA/NCEI's
[GOES Flare Report ReadMe, Appendix A, Table 5](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf).
For every minute, a valid scheduled primary sample is preferred. If it is
unavailable or fails quality filtering, a valid sample from that minute on the
scheduled secondary is used. The generator never uses flare-catalog satellite
IDs to choose the XRS series. If both are unavailable, the minute remains
missing and is not interpolated.

NOAA/NCEI's direct annual file layout depends on the GOES generation:

| Satellites | Official directory and direct-file pattern | Examples |
| --- | --- | --- |
| GOES-8–15 | [NCEI science/XRS archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/) — `goes{N}/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g{N}_y{YYYY}_v2-2-1.nc` | [GOES-15, 2012](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g15_y2012_v2-2-1.nc) (primary); [GOES-14, 2012](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes14/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g14_y2012_v2-2-1.nc) (secondary) |
| GOES-16–19 | [NOAA/NCEI GOES-R archive](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/) — `goes{N}/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g{N}_y{YYYY}_v2-2-1.nc` | [GOES-16, 2023](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g16_y2023_v2-2-1.nc) (primary); [GOES-18, 2023](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g18_y2023_v2-2-1.nc) (secondary) |

The data product version is pinned to `v2-2-1`. Run metadata records every
annual file's exact URL, local cache path, byte size, SHA-256, and valid-sample
count, together with the NOAA schedule source and table.

## Reproducible generation

From the repository root, using the project environment:

```bash
conda run -n rxfi-benchmark python scripts/data/generate_xrs_only_targets.py
```

The script downloads the frozen split indexes from the SuryaBench Hugging Face
dataset and missing NOAA annual files. It records the resolved Hugging Face
commit, all four split-file URLs and checksums, NOAA URLs and checksums, target
definitions, and satellite schedule in its run metadata. Pass
`--revision <40-character-commit-sha> --offline` to regenerate from cached
inputs only. Per-window QC contains valid counts, coverage fractions,
primary/secondary counts, the background estimate, and target availability.

## Role of the flare catalog

The NOAA flare catalog is not an input to these four XRS-only regression
targets. A separately defined `max_old_rxfi` label uses event-level legacy
RXFI for target comparison; its construction and split outputs are specified
in [old_rxfi_split_targets.md](old_rxfi_split_targets.md). Catalog events may
also support operational-skill evaluation on held-out test data.
