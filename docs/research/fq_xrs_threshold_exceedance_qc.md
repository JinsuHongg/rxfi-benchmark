# FQ XRS Threshold-Exceedance Quality Control

## Purpose

This read-only QC asks how often an event-catalog no-flare (`FQ`) forecast window nevertheless contains elevated NOAA/NCEI XRS-B one-minute irradiance. It tests the interpretation of the continuous `max_peak_flux` target; it neither relabels samples nor changes the frozen split files, the confirmed `[t, t + 24h)` contract, or target-generation logic.

## FQ definition and flux thresholds

`max_flare_class = FQ` means that no eligible NOAA flare-summary event peak occurs in `[t, t + 24h)`. It is an event-catalog property, not a statement that every one-minute XRS-B measurement remains below a flare-class boundary. For FQ rows, `max_peak_flux` is the maximum valid science-quality one-minute XRS-B irradiance (W/m2) in the same half-open window.

The exact physical thresholds used here are A1 = `1e-8`, B1 = `1e-7`, C1 = `1e-6`, M1 = `1e-5`, and X1 = `1e-4` W/m2. They are interpretation thresholds only; no categorical class is inferred from them.

## Main results

Of 27,664 FQ rows, 27,559 (99.62%) have a valid positive one-minute maximum and 105 are missing because the window has no valid one-minute observation. Statistics and percentages below use valid FQ maxima only. The distribution has minimum `1.00e-9`, Q1 `1.43e-8`, median `2.93e-8`, mean `8.18e-8`, Q3 `6.32e-8`, P90 `9.27e-8`, P95 `1.20e-7`, P99 `9.53e-7`, and maximum `3.27e-5` W/m2.

| Threshold | FQ windows | Percent of valid FQ windows |
| --- | ---: | ---: |
| >= A1 | 23,762 | 86.22% |
| >= B1 | 2,175 | 7.89% |
| >= C1 | 257 | 0.93% |
| >= M1 | 13 | 0.047% |
| >= X1 | 0 | 0.00% |

The split-wise summary is descriptive only; no modeling choice was made from the test statistics.

| Split | FQ rows | Valid | >= A1 | >= B1 | >= C1 | >= M1 | >= X1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 17,415 | 17,310 | 14,739 (85.15%) | 1,428 (8.25%) | 113 (0.65%) | 10 (0.058%) | 0 (0.00%) |
| Validation | 628 | 628 | 413 (65.76%) | 63 (10.03%) | 17 (2.71%) | 0 (0.00%) | 0 (0.00%) |
| Test | 8,343 | 8,343 | 7,477 (89.62%) | 619 (7.42%) | 122 (1.46%) | 3 (0.036%) | 0 (0.00%) |
| Leaky validation | 1,278 | 1,278 | 1,133 (88.65%) | 65 (5.09%) | 5 (0.39%) | 0 (0.00%) | 0 (0.00%) |

Yearly results show strong temporal clustering, especially in sparsely represented 2012--2015 and 2022--2024 windows. They are descriptive, not a causal satellite- or solar-cycle analysis.

| Year | FQ valid | >= B1 | >= C1 | >= M1 |
| --- | ---: | ---: | ---: | ---: |
| 2010 | 1,150 | 20.87% | 0.26% | 0.00% |
| 2011 | 81 | 60.49% | 6.17% | 1.23% |
| 2012 | 109 | 100.00% | 28.44% | 0.00% |
| 2013 | 101 | 100.00% | 20.79% | 0.00% |
| 2014 | 66 | 100.00% | 66.67% | 13.64% |
| 2015 | 141 | 100.00% | 10.64% | 0.00% |
| 2016 | 814 | 25.06% | 1.60% | 0.00% |
| 2017 | 2,889 | 7.89% | 0.03% | 0.00% |
| 2018 | 6,423 | 4.61% | 0.02% | 0.00% |
| 2019 | 7,442 | 1.64% | 0.01% | 0.00% |
| 2020 | 6,115 | 3.57% | 0.05% | 0.00% |
| 2021 | 2,026 | 9.82% | 0.00% | 0.00% |
| 2022 | 93 | 100.00% | 24.73% | 1.08% |
| 2023 | 43 | 100.00% | 93.02% | 2.33% |
| 2024 | 66 | 100.00% | 84.85% | 1.52% |

## High-flux FQ audit

The complete machine-readable audit contains all 257 C1-or-higher windows and its 13 M1-or-higher subset. For each it records the exact one-minute maximum timestamp, selected satellite, coverage, and nearest NOAA flare-summary events before the forecast window and near the maximum.

- 149/257 C1-or-higher windows have a catalog flare peak within six hours before window start; 65/257 have a catalog event within six hours of the XRS maximum but outside the exact `[t, t+24h)` event-selection window. These timing indicators overlap; their union covers 165/257 cases. The remaining 92 are `unresolved` by these deliberately narrow timing rules.
- 9/13 M1-or-higher windows have a prior peak within six hours and 10/13 have an event near the maximum but outside the forecast window; their union covers 10/13. Three remain `unresolved` by timing alone.

The deterministic local-series audit covers the first 10 C1-or-higher windows ordered by `(timestamp, split)` plus all M1-or-higher windows (22 cases after deduplication). Its heuristic labels are 15 elevated-background, 5 decay-from-prior-flare, 1 broad-gradual-structure, and 1 data-artifact-or-gap. These are descriptions of the ±60-minute trace, not causal determinations. In particular, an `elevated_background` label must not be read as proof that a prior flare caused the value.

This evidence supports the expected semantic distinction: a flare peak just before `t`, or just beyond `t+24h`, can leave an elevated one-minute signal inside an FQ window without violating the event-catalog definition. The timing audit does not establish the mechanism for every case, and it does not diagnose possible catalog incompleteness; those 92 C-level cases and 3 M-level cases remain explicitly unresolved.

## Data-quality checks

The high-flux values are selected only from finite, positive XRS-B observations with science-quality `xrsb_flag == 0`; thus an included maximum has no nonzero quality flag under the frozen validity rule. C1-or-higher windows have median coverage 0.9778 and minimum 0.0417; M1-or-higher windows have median coverage 0.9722 and minimum 0.6757. All 257 C-level and all 13 M-level maxima used the scheduled primary source (no secondary fallback). Satellite counts for C-or-higher are GOES-13: 3, GOES-14: 3, GOES-15: 127, GOES-16: 1, GOES-17: 27, GOES-18: 96; M-or-higher are GOES-15: 10, GOES-17: 1, GOES-18: 2.

The low minimum C-level coverage warrants case-level caution, but the M-level cases do not show a similarly sparse-window pattern. No target was altered by this audit: frozen split SHA-256 values match the confirmed contract, derived row counts remain unchanged, and the script verifies identical derived-target SHA-256 values before and after analysis.

## Reproducible artifacts

- [Overall threshold summary](../../outputs/fq_xrs_threshold_exceedance_qc/fq_threshold_exceedance_overall.csv)
- [Split and yearly summaries](../../outputs/fq_xrs_threshold_exceedance_qc/fq_threshold_exceedance_by_split.csv) and [yearly CSV](../../outputs/fq_xrs_threshold_exceedance_qc/fq_threshold_exceedance_by_year.csv)
- [C-level audit](../../outputs/fq_xrs_threshold_exceedance_qc/fq_c1_or_higher_audit.csv), [M-level audit](../../outputs/fq_xrs_threshold_exceedance_qc/fq_m1_or_higher_audit.csv), and [local-series cases](../../outputs/fq_xrs_threshold_exceedance_qc/local_timeseries_case_summary.csv)
- [Flux distribution with thresholds](../../outputs/fq_xrs_threshold_exceedance_qc/figure_fq_flux_distribution_thresholds.svg), [split exceedance](../../outputs/fq_xrs_threshold_exceedance_qc/figure_fq_threshold_exceedance_by_split.svg), and [yearly exceedance](../../outputs/fq_xrs_threshold_exceedance_qc/figure_fq_threshold_exceedance_by_year.svg)

The event-proximity audit used the NOAA/NCEI science-quality Flare Report mission file retrieved 2026-09-15 (`sci_xrsf-l2-flrpt_geo_s19950103_e20260914_v1-0-1.csv`); its SHA-256 is recorded in [metadata.json](../../outputs/fq_xrs_threshold_exceedance_qc/metadata.json). NOAA's [Flare Report directory](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/) is live and may later revise releases.

## Conclusion

**PROCEED WITH DOCUMENTED CAVEAT.** A/B-level exceedance is common, as expected for a continuous irradiance maximum. C-level exceedance is uncommon overall (0.93%), M-level exceedance is very rare (0.047%), and no X-level FQ maximum occurs. Most M-level and many C-level examples have a nearby catalog-event timing explanation; the remaining cases are retained as unresolved QC findings, not silently reclassified. Therefore `log10(max_peak_flux)` remains a defensible continuous target across FQ and flare windows for OCQR, provided the event-catalog/instantaneous-flux distinction and the unresolved high-FQ cases are documented.
