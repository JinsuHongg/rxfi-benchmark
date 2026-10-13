# Paper-Based 24-Hour Flare Background: Upper-Cap Comparison

## Scope and inputs

This comparison applies the procedure in Hong et al., Section 2.1, to NOAA/NCEI science-quality GOES flare events with peak times from 2010-01-01 through 2025-12-31. The event catalog is the mission Flare Report release `sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv` (SHA-256 `351eda39efea3f5eb4d394b91956d1c08e7f63993c41d09d686e5ad2983cbbea`). The measurement inputs are the satellite-specific science-quality XRS L2 `avg1m` NetCDF files, version `v2-2-1`, using `xrsb_flux` and `xrsb_flag`. The exact URLs, byte sizes, retrieval status, and SHA-256 for all 43 measurement files are recorded in [input_manifest.json](../../outputs/paper_background_sensitivity/input_manifest.json); the downloaded files are cached under the ignored `data/raw/noaa_paper_background/` directory.

The cohort contains 41,723 events from GOES-13, -14, -15, -16, -17, -18, and -19. Seven exact `(flare_id, time)` duplicate catalog rows were removed. All retained events have valid start/end times, source satellite, and positive finite peak flux.

## Calculation

For each event with start time (t_0), candidate measurements come from the same satellite in the strictly pre-start 24-hour interval ([t_0-24\mathrm{h},t_0)), in UTC. A measurement is valid when its XRS-B flux is finite and positive and `xrsb_flag == 0`. Samples at timestamps inside another catalog flare's inclusive start/end interval are excluded.

The uncapped background is the arithmetic mean of all remaining valid samples. The capped background uses the same candidate pool, additionally retaining only samples (X(t)\le X(t_0)), where (X(t_0)) is the exact valid same-satellite XRS L2 observation at the catalog start time. The start observation is not interpolated or replaced by a nearby sample. If it is missing or invalid, the capped value is missing. If no pre-start sample remains below the cap, the capped value is also missing.

For consistency with the project's selected convention, event RXFI is calculated as the fractional excess ((F_{peak}-B)/B). This is one less than the paper's stated peak-to-background ratio (F_{peak}/B). NOAA's event-level `background_irrad` is retained for comparison only.

## Results

| Quantity | No upper cap | With upper cap |
| --- | ---: | ---: |
| Valid event backgrounds | 41,723 | 41,220 |
| Median background (W/m²) | (9.28\times10^{-7}) | (6.76\times10^{-7}) |
| Mean background (W/m²) | (1.42\times10^{-6}) | (9.08\times10^{-7}) |
| Median valid-minute coverage | 86.9% | 47.2% |
| Median fractional-excess RXFI | 0.777 | 1.376 |
| Negative fractional-excess RXFI | 4,842 | 0 |

Among the 41,220 events with both estimates, the capped background is lower in every case or equal, with a median relative change of -23.3% and a mean change of -26.3%. The cap removes 24,425,599 event-window samples in total (sample counts recur across overlapping event windows); 27,358,751 samples remain after the cap. The median capped-minus-uncapped RXFI change is +0.575. Paired capped and uncapped RXFI values have Spearman correlation 0.878.

There are 5 events without a usable exact start-time observation (4 missing and 1 invalid-quality/flux), so their capped estimates are unavailable. Of the remaining events, 498 have a valid start threshold but no valid pre-start sample at or below it. The uncapped windows have a median of 1,252 valid minutes (out of 1,440); the capped windows have a median of 679. These coverage figures reflect product gaps, quality screening, other-flare masks, and the upper cap.

The uncapped background exceeds the catalog peak for 4,842 events, producing negative fractional excesses; these values are retained. No capped background exceeds the peak in this cohort, consistent with the upper-filter rule for this data. The two methods therefore differ materially in both background level and RXFI scale, not just in a handful of outliers.

## Reproducible outputs

- [Event-level results](../../outputs/paper_background_sensitivity/event_backgrounds.csv): NOAA event data, both backgrounds and RXFI values, satellite, start-time threshold status, valid and masked sample counts, coverage, and per-variant status.
- [Input manifest](../../outputs/paper_background_sensitivity/input_manifest.json): NOAA release identifiers, URLs, sizes, retrieval indicators, and checksums.
- [Run summary](../../outputs/paper_background_sensitivity/summary.json): cohort and computation counts.
- [Statistical comparison](../../outputs/paper_background_sensitivity/analysis_summary.json): quantiles, paired differences, correlations, coverage, and NOAA-field comparison.
- [Calculation script](../../scripts/background/compute_paper_background_sensitivity.py): verifies the pinned inputs and recreates results in a new fingerprinted run directory. Use [the run guide](../../outputs/paper_background_sensitivity/README.md) for offline/online commands and environment setup.

The run guide documents one legacy boundary behavior retained to reproduce these frozen rows: flare intervals are selected by their UTC start year among the calendar years touched by the event window. An interval beginning in the prior year but overlapping a window can therefore be missed at New Year's. This is not specified by the paper and should be addressed only in an explicitly versioned future recalculation.

The frozen row-level forecast split files are not present in this checkout (`/mnt/storage/surya/index_data` is unavailable), so this work compares event backgrounds and does not regenerate or aggregate project window targets. Existing target files were not changed.

## Interpretation

The upper cap substantially lowers the estimated background and raises RXFI while removing about half of the valid 24-hour samples for a typical event. It also guarantees that the mean background does not exceed the start threshold, which eliminates negative RXFI in this cohort. That is a property of the filter and should not be interpreted as evidence that the uncapped estimates are erroneous: uncapped values retain elevated preflare activity and are useful for measuring sensitivity to the filter. The two variants should remain separately labeled in downstream analysis.

Source method: Hong et al., [*Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*](https://par.nsf.gov/servlets/purl/10435332), Section 2.1. Product details: NOAA/NCEI [GOES-R XRS L2 Data User's Guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf).
