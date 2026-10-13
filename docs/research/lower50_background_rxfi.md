# Lower-50% 24-hour X-ray background and RXFI

## Decision and interpretation

This project variant estimates each event's background from the 24 hours immediately before its catalog start time. It retains positive, finite GOES XRS-B `avg1m` samples with science-quality flag `xrsb_flag == 0`, then removes samples within the catalog start-to-end interval of another flare. From the remaining `N` measurements, it sorts flux from low to high and averages the lowest \(\lceil 0.5N\rceil\) observations:

\[
B_{50} = \operatorname{mean}\left(\operatorname{lowest}_{\lceil N/2\rceil}(X_1,\ldots,X_N)\right).
\]

For odd `N`, the number selected is rounded up. There is no minimum-flux truncation and no start-time upper cap in `B_50`. The `POST_EVENT` flag is not used. Other-flare intervals are excluded using the NOAA flare report's catalog start and end times.

The project defines relative increase as fractional excess above background:

\[
\mathrm{RXFI}_{50} = \frac{F_{\mathrm{peak}} - B_{50}}{B_{50}}.
\]

The runner also retains the existing full-mean and start-capped mean columns as comparisons; the lower-half columns are the requested new estimator.

The event-level result preserves negative RXFI values as calculated for provenance. The forecast-window target generator applies an additional target rule: events with RXFI below zero are excluded from `max_rxfi` and `cumulative_rxfi`, but remain eligible for peak-flux targets. If a window has no non-negative RXFI event, both RXFI targets are set to zero. The current 2010–2025 event table contains 261 such negative values.

## Relationship to the paper

Hong et al., *Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*, describes a filtered 24-hour arithmetic mean and excludes low-quality points, measurements in other flare intervals, and measurements above the target event's start-time flux. The lower-50% mean is a **project adaptation** that further reduces the influence of residual elevated flux after catalog intervals are removed. It is not the estimator specified by that paper and should be evaluated as a separate method. Source: [NSF Public Access Repository paper](https://par.nsf.gov/servlets/purl/10435332), Section 2.1.

## Inputs and reproducibility

The calculation uses the pinned NOAA/NCEI science-quality flare report and GOES XRS Level 2 `avg1m` files documented by [`configs/background_lower50.json`](../../configs/background_lower50.json) and [`configs/paper_background_sensitivity.inputs.lock.json`](../../configs/paper_background_sensitivity.inputs.lock.json). It selects the GOES satellite specified for each catalog event and uses the project's locked event cohort and time boundaries. Input checksums and exact run configuration are written with the results.

## Recalculation result

The current offline run used all locked local inputs and produced 41,723 events from 2010-01-01 through 2025-12-31. All 41,723 events had an eligible `B_50` and RXFI. The median `B_50` was `6.767769e-7 W m^-2`; median RXFI was `1.42337` and the 90th percentile was `7.51969`. There were 3,022 M-class events (median RXFI `10.1785`) and 167 X-class events (median RXFI `105.233`).

No event had an uncomputable lower-50% RXFI in this run. RXFI was negative for 261 events (0.63%): 2 A-class, 143 B-class, and 116 C-class events; no M/X-class event was negative. A negative value occurs when the catalog peak flux is below the estimated pre-event background. These event-level values remain in the background result table, while target aggregation excludes them as described above.

Across 41,721 events with a positive NOAA catalog `background_irrad`, the median ratio `B_50 / B_NOAA` was `0.8348`; 30,799 project estimates were below the catalog value and 10,922 above it. This comparison is descriptive: the two background values use different estimators.

The calculation retains a legacy interval lookup detail for reproducibility: other-flare intervals are indexed by their UTC start year within the year or years touched by the target's 24-hour window. A flare that started in the preceding year but overlaps a window after New Year's may therefore not be masked. This boundary limitation is carried forward from the existing runner.

Run from the repository root in the project environment:

```bash
conda run -n rxfi-benchmark python scripts/background/compute_paper_background_sensitivity.py \
  --config configs/background_lower50.json --offline
```

The output is created in `outputs/background_lower50/runs/<fingerprint>/` and includes `event_backgrounds.csv`, `summary.json`, `input_manifest.json`, and `run_config.json`. `background_lower_fraction_W_m2` and `rxfi_lower_fractional_excess` contain the new background and RXFI; `lower_retained_minutes` and `lower_coverage_fraction` describe how many measurements went into the lower-half mean. Empty RXFI cells indicate that no eligible background samples remained. The current run and summary are listed in [`outputs/background_lower50/README.md`](../../outputs/background_lower50/README.md).
