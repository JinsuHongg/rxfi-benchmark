# Deterministic 24-Hour Flare Target Generation

## Purpose

This document defines four continuous Project-1 targets from the same frozen 24-hour forecasting windows. They express two complementary axes without selecting a downstream learning formulation: absolute versus background-relative magnitude, and maximum versus cumulative activity.

## Frozen sample splits

`train.csv`, `validation.csv`, `test.csv`, and `leaky_validation.csv` under `/mnt/storage/surya/index_data` are immutable inputs. The generator reads them only and writes separate target files under `data/derived/`. Every output contains `original_row_index`, preserves source row order and timestamp exactly, and retains all source columns.

| Split | Source rows | Derived rows | SHA-256 unchanged | Row identity preserved |
| --- | ---: | ---: | --- | --- |
| Train | 74,760 | 74,760 | Yes | Yes |
| Validation | 3,672 | 3,672 | Yes | Yes |
| Test | 43,848 | 43,848 | Yes | Yes |
| Leaky validation | 6,048 | 6,048 | Yes | Yes |

The derived target files are intentionally gitignored because they are reproducible local data products (about 17 MB total), not source code or hand-maintained research artifacts.

## Confirmed forecasting-window contract

For every frozen sample timestamp \(t\), the confirmed dataset contract uses the half-open future window:

\[
[t, t + 24\ \mathrm{hours}).
\]

All NOAA flare events whose peak time lies in this interval are eligible. An event exactly at \(t+24\) is excluded; a built-in assertion verifies this `bisect_left` boundary behavior. Windows may cross day, month, and year boundaries. The 2025 NOAA data present in the mission file provide full 24-hour coverage for 2024-12-31 test timestamps.

## Event-level quantities

For a NOAA flare event \(i\), valid peak flux is its finite, positive XRS-B `xrsb_irrad` value \(F_i\), in W/m2. Event-level RXFI reuses the shared frozen implementation from `scripts/inspect_noaa_xrs_flare_report.py`:

\[
\mathrm{RXFI}_i = \frac{F_i - B_i}{B_i},
\]

where \(B_i\) is NOAA `background_irrad`. Peak-flux aggregation accepts all events with valid \(F_i\); RXFI aggregation accepts only events with valid RXFI. The source CSV's `flare_id` is stored as the event identifier; if it is blank, the deterministic fallback key is `time|xrsb_irrad_source`.

## Sample-level targets

For valid-peak events \(E_t\) in a sample window and valid-RXFI events \(R_t \subseteq E_t\), the generated fields are:

\[
\begin{aligned}
\texttt{max_peak_flux} &= \max_{i \in E_t} F_i, \\
\texttt{cumulative_peak_flux} &= \sum_{i \in E_t} F_i, \\
\texttt{max_rxfi} &= \max_{i \in R_t} \mathrm{RXFI}_i, \\
\texttt{cumulative_rxfi} &= \sum_{i \in R_t} \mathrm{RXFI}_i.
\end{aligned}
\]

`max_flare_class` is the NOAA `flare_class` of the event selected by `max_peak_flux`; it is categorical. All four aggregate quantities remain continuous and are not binned. `flare_count_24h` counts valid-peak events. `max_peak_flare_id` and `max_rxfi_flare_id` retain the selected identifiers independently because they can identify different events.

## No-flare and tie conventions

If a window has no valid-peak events, `flare_count_24h = 0`, all four continuous targets are zero, both identifier fields are empty, and `max_flare_class = FQ`. If a window has peak-flux events but none with valid RXFI, peak targets are still aggregated while both RXFI targets are zero and `max_rxfi_flare_id` is empty.

Ties are not discarded. For maximum peak flux and maximum RXFI separately, the deterministic winner is the earliest peak time; remaining ties use lexical stable event ID order. The generator reports the number of tied candidate windows.

## Quality-control results

The mission catalog produced 41,723 valid-peak events over the loaded 2010-2025 coverage. Its duplicate stable-key and invalid-peak counts are recorded in [target_generation_metadata.json](../../outputs/target_generation_24h/target_generation_metadata.json). Across sample windows, invalid-RXFI event occurrences were 48 in train and zero in the other three splits; these are retained for peak aggregation and excluded only from RXFI aggregation.

| Split | No-flare windows | Multi-flare windows | Different peak/RXFI event, among comparable windows | Peak ties |
| --- | ---: | ---: | ---: | ---: |
| Train | 23.29% | 71.30% | 25.73% | 35 |
| Validation | 17.10% | 77.72% | 23.82% | 0 |
| Test | 19.03% | 76.38% | 25.68% | 40 |
| Leaky validation | 21.13% | 72.24% | 26.33% | 0 |

The exact numeric results, including complete target statistics, quality counts, and mismatch examples, are written to `outputs/target_generation_24h/`.

## Validation against existing labels

Exact `max_flare_class` string agreement with the existing `max_goes_class` labels is 21.29% (train), 15.03% (validation), 80.54% (test), and 19.49% (leaky validation). These comparisons are retained as validation results and do not overwrite the existing labels.

The dominant mismatch pattern in the older splits is a science-quality NOAA class with a different numeric magnitude than the supplied historical label (for example, `B2.8` versus `B3.7`, or `B8.3` versus `C1.1`). This is consistent with the earlier documented calibration/version distinction between historical operational labels and NOAA's reprocessed science-quality catalog. Test mismatches include windows labelled `A1.0` in the source but with no valid NOAA flare-report event; the report's catalog threshold/coverage must be considered when interpreting those rows. The target-generation contract itself is confirmed; this comparison evaluates compatibility with the legacy labels, not split membership or event-window selection.

## Target summaries and figures

`target_summary_by_split.csv` reports valid count, zero count, mean, median, standard deviation, Q1/Q3, p90/p95/p99, and maximum for each target and split. The two proposal-oriented figures use `log10(1 + value)` only for display and retain zero/no-flare windows:

- [Four target distributions](../../outputs/target_generation_24h/figure_24h_target_distributions.svg)
- [Maximum peak flux versus maximum RXFI](../../outputs/target_generation_24h/figure_max_peak_flux_vs_max_rxfi.svg)

## Project-1 interpretation and open questions

The targets deliberately span two conceptual axes: `max_peak_flux`/`cumulative_peak_flux` measure absolute X-ray activity, whereas `max_rxfi`/`cumulative_rxfi` measure background-relative activity; maximum and cumulative forms distinguish a dominant event from total 24-hour activity. This generation work makes no superiority claim and does not select target bins, model architecture, or uncertainty-quantification design.

Open questions for the next stage are restricted to downstream experimental design: whether to predict continuous or transformed targets, how to handle the documented legacy-label compatibility differences, and what validation-only target transformations—if any—are justified.
