# Paper-Based Background Cap Comparison Plan

> **For agentic workers:** Use `superpowers:executing-plans` to carry out this plan task by task. Do not change the existing target-generation outputs during this comparison.

**Goal:** Calculate event-level paper-based background irradiance and fractional RXFI both with and without the paper's start-flux upper filter, then quantify the impact on the project's frozen 24-hour labels.

**Architecture:** Use the NOAA/NCEI science-quality GOES Flare Report as the event catalog and the NOAA/NCEI science-quality XRS L2 one-minute series as the measurement source. Calculate both background variants from the same event, satellite, time window, quality-screened measurements, and other-flare mask; the only difference is whether measurements above the target flare's start-time flux are removed. Save new, provenance-pinned comparison outputs separately from existing targets.

**Tech Stack:** Python, CSV, NOAA NetCDF XRS L2 `avg1m`, NumPy/xarray or the repository's existing NetCDF reader, JSON metadata.

**Spec:** `docs/research/paper_based_background_xray_flux_method.md`

## Global Constraints

- Use NOAA science-quality GOES Flare Report events and science-quality XRS L2 `avg1m` data.
- Study events are catalog events with peak times from 2010-01-01 through 2025-12-31; include the prior day of XRS data needed for the first event windows.
- Match a flare's one-minute series by its `xrsb_irrad_source`; do not average different satellites or silently substitute another satellite.
- For flare start `t0`, use the strictly pre-start interval `[t0 - 24 hours, t0)` in UTC.
- Use finite positive XRS-B measurements with the NOAA quality flag equal to zero; mask measurements inside another catalog flare's inclusive `[start_time, end_time]` interval.
- Upper-filter case retains only measurements `X(t) <= X(t0)`, where `X(t0)` is the exact, valid same-satellite L2 observation at the catalog `start_time`.
- No-cap case applies the same rules but does not filter by `X(t0)`.
- Calculate project fractional excess as `(F_peak - B) / B` using catalog `xrsb_irrad` as `F_peak`; keep negative valid values in the no-cap case and do not clamp them.
- Do not overwrite or regenerate existing derived targets, train models, or change model configurations in this comparison task.
- Record source URLs, retrieval time, product versions, and SHA-256 checksums in output metadata.

## Review Focus

- Missing or low-quality observation exactly at `start_time`: upper-filter background is missing; do not interpolate or substitute a nearby sample. Still calculate the no-cap background when possible and report the unmatched count.
- Satellite-source coverage gaps in the preceding day: report available and retained minute counts and coverage; never bridge gaps by mixing satellites.
- NOAA flare intervals overlapping the background window: exclude their inclusive start/end samples and record the number of masked intervals and samples.
- Uncapped background exceeding peak: retain the resulting negative fractional excess as a valid diagnostic when both `F_peak` and `B` are valid and positive.
- Updated NOAA releases: pin exact catalog and L2 file versions/checksums so a rerun cannot silently mix releases.

---

### Task 1: Pin the NOAA inputs and define the event cohort

**Files:**
- Create: `scripts/compute_paper_background_sensitivity.py`
- Create: `outputs/paper_background_sensitivity/input_manifest.json`

- [ ] Retrieve the latest available NOAA/NCEI science-quality mission Flare Report and use events with peak times in 2010–2025. At plan time, NOAA lists `sci_xrsf-l2-flrpt_geo_s19950103_e20261011_v1-0-1.csv`; confirm availability and record the exact filename, URL, retrieval date, and SHA-256 before processing.
- [ ] Read `start_time`, `end_time`, `time`, `flare_id`, `xrsb_irrad`, `xrsb_irrad_source`, `flare_class`, and `background_irrad`; retain NOAA `background_irrad` for the comparison table only.
- [ ] Deduplicate only exact `(flare_id, time)` duplicates, and report duplicate counts and events with missing timing, source, or valid peak.
- [ ] Retrieve needed science-quality satellite-specific XRS L2 `avg1m` files for the event-source satellite and 24-hour pre-start periods. Pin each product version and checksum. Do not fetch unrelated time ranges if daily/monthly file selection can cover the event windows.
- [ ] Emit an input manifest listing each source file, product version, URL, size, retrieval time, and SHA-256.

**Checkable result:** the event table and manifest identify a reproducible NOAA source release and the full event cohort.

### Task 2: Calculate paired event-level backgrounds

**Files:**
- Modify: `scripts/compute_paper_background_sensitivity.py`
- Create: `outputs/paper_background_sensitivity/event_backgrounds.csv`

- [ ] For each event, select its exact satellite's XRS-B one-minute observations in `[start_time - 24h, start_time)` using UTC product timestamps.
- [ ] Keep finite, positive samples with the science-quality flag set to zero. Exclude samples whose timestamps fall within any other Flare Report event's inclusive start/end interval.
- [ ] Read the upper-filter threshold only from the exact, valid same-satellite sample at the target `start_time`. If that sample is absent or invalid, set the capped estimate to missing and record the reason; do not interpolate.
- [ ] Compute `B_uncapped` as the arithmetic mean of the common valid, non-flare samples. Compute `B_capped` as the arithmetic mean of that same pool after applying `X(t) <= X(t0)`.
- [ ] For each valid background, calculate `RXFI_uncapped = (xrsb_irrad - B_uncapped) / B_uncapped` and `RXFI_capped = (xrsb_irrad - B_capped) / B_capped`. Preserve negative uncapped values.
- [ ] Include per-event provenance and diagnostics: source satellite, start/peak times, catalog peak, NOAA `background_irrad`, start threshold, candidate sample count, counts removed by each rule, retained sample counts, and coverage fractions.

**Checkable result:** one event-level row per catalog event, with both paired estimates where computable and explicit missingness reasons otherwise.

### Task 3: Aggregate both variants onto the frozen project windows

**Files:**
- Modify: `scripts/compute_paper_background_sensitivity.py`
- Create: `outputs/paper_background_sensitivity/window_targets.csv`

- [ ] Read the existing frozen split timestamps without changing membership, order, or source files.
- [ ] Associate events by peak time in the existing half-open forecast interval `[t, t + 24h)`.
- [ ] For each window, separately compute maximum and cumulative fractional RXFI using the capped and uncapped event-level values. Keep event counts and selected flare IDs per variant.
- [ ] Preserve the existing no-flare convention as zero for window-level RXFI targets; keep no-flare windows distinguishable using `flare_count_24h`.
- [ ] Match the existing target-generation convention: aggregate events with valid RXFI for each variant, report valid and invalid-RXFI event counts per window, and use zero when no event has valid RXFI. Do not silently drop invalid events from the QC counts.

**Checkable result:** a side-by-side capped/uncapped target table aligned row-for-row with frozen splits, with no existing target file modified.

### Task 4: Summarize the cap's effect and document the comparison

**Files:**
- Create: `outputs/paper_background_sensitivity/summary.json`
- Create: `docs/research/paper_background_cap_sensitivity.md`
- Modify: `docs/research/current_state.md`

- [ ] Report cohort counts, successful/missing background counts, coverage summaries, and the number/fraction of candidate samples removed only by the upper filter.
- [ ] For the paired event cohort, summarize `B_capped - B_uncapped`, fractional change in background, `RXFI_capped - RXFI_uncapped`, rank correlation, quantiles, and counts where uncapped RXFI is negative.
- [ ] For frozen windows, report paired differences for `max_rxfi` and `cumulative_rxfi`, with counts of windows whose selected maximum event changes.
- [ ] Compare both computed backgrounds descriptively against NOAA `background_irrad`; do not treat NOAA's event field as ground truth for the paper-based 24-hour estimator.
- [ ] Include methods, input versions/checksums, conventions, exclusions, coverage limitations, and all counts in the research note and metadata.
- [ ] Run repository whitespace/diff checks and verify that pre-existing target files and split files remain unchanged.

**Checkable result:** a reviewable report quantifies whether the upper cap materially changes backgrounds, RXFI values, and project-window targets.

## Source references

- Hong et al., Section 2.1: [Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions](https://par.nsf.gov/servlets/purl/10435332).
- NOAA/NCEI Flare Report CSV release directory: [science-quality GOES Flare Report](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/).
- NOAA/NCEI XRS L2 guidance: [GOES-R XRS L2 Data User's Guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf).
- NOAA/NESDIS/STAR: [GOES-R XRS.07 Event Detection Algorithm Theoretical Basis Document](https://www.ngdc.noaa.gov/stp/space-weather/online-publications/stp_sii/spades/algorithm-theoretic-basis-documents/atbds_exis/atbd_xrs07_event-detection_v1-2.4.pdf).
