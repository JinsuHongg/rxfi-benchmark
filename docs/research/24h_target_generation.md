# Legacy: Catalog-based 24-hour Flare Target Generation

> **Status:** This document describes the existing legacy generator and its
> catalog-based split targets. The approved research targets are now defined
> from XRS-only data in [xrs_only_target_definitions.md](xrs_only_target_definitions.md).
> The XRS-only targets have not yet been implemented or regenerated.

## Source splits and provenance

The target generator reads the four published CSV files from the [NASA-IBM SuryaBench flare-forecasting dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting/tree/main). Their temporal membership follows the dataset's split policy: train covers February 15–December 31 in 2010–2019; validation covers January 15–31 in 2010–2019; leaky validation covers January 1–14 and February 1–14 in 2010–2019; test covers 2020–2024. The retrieved source files contain 128,328 rows, matching the documented split sizes.

| Split | Source rows | SHA-256 |
| --- | ---: | --- |
| Train | 74,760 | `2f3a40b2074ac8984593d53df4f81434ec91f34bb7629066252f3e0bcd8e37b7` |
| Validation | 3,672 | `b4e29566e786c45693a5d78eef73c1f05f6e19fb87f6e2f255dbfa5b81cb256d` |
| Test | 43,848 | `158901e21da5a87720b5db1200dbd4765b7ad7dc692157f9bf87f2e8eeefdf58` |
| Leaky validation | 6,048 | `dfe13927a3afc0780831100e5810ae29f750acbe7e951338dada3ecef3da68a6` |

Checksums identify the exact retrieved files used by this run. The generator does not change its inputs; each target CSV preserves source columns, row order, and timestamps and appends `original_row_index` plus generated target columns. The source split files and derived targets are local, gitignored data products.

## Forecast window and event source

For each sample timestamp (t), the forecast window is the half-open interval:

\[
[t,t+24\ \mathrm{hours}).
\]

An event at exactly (t+24\) hours is excluded. The generator uses finite, positive NOAA/NCEI GOES XRS Flare Report `xrsb_irrad` peaks whose catalog peak time falls in the window. The source catalog is pinned in [`configs/background_lower50.json`](../../configs/background_lower50.json). The full event-level lower-50% background results are joined by catalog peak time and `flare_id`; the generator also checks that the event peak values agree.

## Event background and RXFI

For an event with catalog start time \(t_{\mathrm{start}}\), the revised project background \(B_{50}\) is calculated from same-satellite GOES XRS Level 2 `avg1m` XRS-B measurements in \([t_{\mathrm{start}}-48\,\mathrm{h},t_{\mathrm{start}}-24\,\mathrm{h})\). The event-level calculation keeps positive, finite samples with `xrsb_flag == 0`, removes samples within another catalog flare's inclusive start–end interval, sorts the remaining (N) samples by flux, and takes the arithmetic mean of the lowest \(\lceil N/2\rceil\):

\[
B_{50}=\operatorname{mean}(\operatorname{lowest}_{\lceil N/2\rceil}(X_1,\ldots,X_N)).
\]

RXFI is the fractional increase above this background:

\[
\mathrm{RXFI}_{50}=\frac{F_{\mathrm{peak}}-B_{50}}{B_{50}}.
\]

This is a project adaptation of the background discussion in Hong et al., [*Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*](https://par.nsf.gov/servlets/purl/10435332), Section 2.1. It is not the paper's full arithmetic-mean estimator: it uses the lower half to reduce residual elevated measurements after other-flare masking, does not apply the paper's start-time upper cap, and does not use `POST_EVENT`. NOAA `background_irrad` is not used in the current split-level RXFI targets.

**Implementation status:** the revised interval above is a documentation/method-definition change. The current generator, event-background table, and split target CSVs still use the previous interval \([t_{\mathrm{start}}-24\,\mathrm{h},t_{\mathrm{start}})\); the event backgrounds and targets must be regenerated before they can be described as outputs of the revised definition. The FQ pseudo-event rule remains separately anchored to its in-window maximum and has not been changed here.

The current background run contains 41,723 events from 2010-01-01 through 2025-12-31. All 41,723 have a valid lower-50% background and RXFI. The event-level table preserves negative RXFI values for provenance; 261 are negative because their peak flux is below the estimated background. For forecast-window target generation, those negative events are excluded from RXFI aggregation while remaining eligible for the peak-flux targets. See [lower50_background_rxfi.md](lower50_background_rxfi.md) for event-level calculation details and limitations.

## Generated targets

Let `E_t` be the valid-peak events and `R_t` the events with valid, non-negative RXFI in a forecast window:

\[
\begin{aligned}
\texttt{max\_peak\_flux} &= \max_{i\in E_t} F_i, \\
\texttt{cumulative\_peak\_flux} &= \sum_{i\in E_t} F_i, \\
\texttt{max\_rxfi} &= \max_{i\in R_t} \mathrm{RXFI}_{50,i}, \\
\texttt{cumulative\_rxfi} &= \sum_{i\in R_t} \mathrm{RXFI}_{50,i}.
\end{aligned}
\]

`max_flare_class` is the catalog class of the maximum-peak event. `flare_count_24h` counts valid-peak events. `max_peak_flare_id` and `max_rxfi_flare_id` identify their respective maximum events, which can differ. Ties are resolved by earliest peak time, then lexical event ID.

### FQ regression targets

`FQ` means that the forecast window contains no valid NOAA flare-catalog peak. It does not mean that XRS-B measurements are zero. For regression, an FQ window is assigned one **window-maximum pseudo-event** from the science-quality XRS-B one-minute series:

1. In `[t, t+24 h)`, select the largest finite, positive `xrsb_flux` measurement with `xrsb_flag == 0`. Let its timestamp be `a` and its flux be `F*_t`. If multiple samples tie, use the earliest timestamp.
2. Estimate its background from `[a - 24 h - 5 min, a - 5 min)`, using the same-satellite XRS-B samples, quality filter, other-catalog-flare interval masking, and lower-50% mean defined above. The 5-minute offset is a project-selected buffer before the selected maximum; it is not asserted to be a universal NOAA detection delay.
3. Calculate the pseudo-event's relative increase as `RXFI*_t = (F*_t - B50(a)) / B50(a)`.
4. Treat this as a one-value event set, so `max_peak_flux = cumulative_peak_flux = F*_t` and, when `RXFI*_t` is non-negative, `max_rxfi = cumulative_rxfi = RXFI*_t`. The existing rule for negative RXFI is retained: a negative value is excluded from RXFI aggregation, leaving both RXFI targets at zero.

If the prediction window has no valid XRS-B samples, its FQ flux and RXFI regression targets are missing rather than imputed as zero. If the peak flux exists but the preceding interval has no eligible samples for `B50`, the two flux targets remain defined and the two RXFI targets are missing. `flare_count_24h` remains zero, `max_flare_class` remains `FQ`, and catalog event IDs remain empty: the pseudo-event is not added to the NOAA event catalog or flare count.

This definition intentionally gives FQ rows continuous regression targets while preserving FQ as a no-catalog-event classification label. For flare-containing windows, the flux targets aggregate NOAA catalog event peaks; for FQ windows, they represent the maximum observed XRS-B measurement as one pseudo-event. The cumulative and maximum values are equal for FQ by definition.

**Implementation status:** the target generator applies this rule to the four local derived split files. The run preserves all 128,328 source rows and their order. Of 27,664 FQ windows, 27,559 have a valid window maximum and 105 have no valid XRS-B sample; 31 additional windows have a valid maximum but no eligible `B50` background samples, so their RXFI targets are missing. Among the remaining windows, 97 calculated RXFI values are negative and excluded, while 10 equal the background exactly; both cases yield zero RXFI targets. The other 27,421 FQ windows have positive RXFI. The run's `target_quality_control.csv` and `target_generation_metadata.json` record per-split counts, XRS input checksums, and target provenance.

For catalog events, a valid peak with unavailable or negative RXFI still contributes to the two peak-flux targets, but not to the RXFI aggregates. If no event remains in `R_t`, both RXFI targets are zero and `max_rxfi_flare_id` is empty. The current event cohort had no missing lower-50% RXFI; 261 negative-RXFI events were excluded from RXFI aggregates. Since forecast windows overlap, per-split QC counts these exclusions once per window occurrence. By construction, the generated catalog-event `max_rxfi` and `cumulative_rxfi` targets are non-negative.

## Current generated artifacts and quality checks

Split-specific target files are in `data/derived/{train,validation,test,leaky_validation}_targets.csv`. Each has the same row count as its source split, and row order and timestamps were verified. Detailed summaries and QC are stored in `outputs/target_generation_24h/`:

- `target_generation_metadata.json`: catalog/background checksums, selected RXFI method, split counts, and source split checksums.
- `target_summary_by_split.csv`: target distribution summaries by split.
- `target_quality_control.csv`: no-flare and multi-flare windows, RXFI exclusions, and tie counts.
- `target_validation_against_existing_labels.csv`: regenerated NOAA class versus the source `max_goes_class` label comparison.
- Two SVG figures showing target distributions and the maximum-flux/RXFI relationship.

The regenerated NOAA class strings do not overwrite source labels. Agreement is 21.29% for train, 15.03% for validation, 80.54% for test, and 19.49% for leaky validation. Earlier-period differences are consistent with the documented distinction between historical labels and the current science-quality catalog; the comparison is a compatibility diagnostic, not a change to split membership.

## Reproduction

Run from the repository root with the `rxfi-benchmark` Conda environment. Download the source splits from the Hugging Face dataset into `data/raw/surya_bench_splits/`. Ensure the checksum-pinned NOAA catalog and required XRS files are present in `data/raw/noaa_paper_background/`, then calculate event backgrounds:

```bash
python scripts/background/compute_paper_background_sensitivity.py \
  --config configs/background_lower50.json --offline
```

Use the resulting `event_backgrounds.csv` path as input to target generation:

```bash
python scripts/data/generate_24h_targets.py \
  --data-dir data/raw/surya_bench_splits \
  --noaa-csv data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv \
  --rxfi-method lower50 \
  --lower50-backgrounds-csv outputs/background_lower50/runs/<fingerprint>/event_backgrounds.csv \
  --derived-dir data/derived \
  --output-dir outputs/target_generation_24h
```

Replace `<fingerprint>` with the directory created by the background calculation. Omit `--offline` from that calculation only if missing pinned inputs need to be downloaded. `--rxfi-method` selects how catalog-event RXFI is calculated; `lower50` explicitly selects the current project method and requires the event-background table. FQ pseudo-event RXFI always uses the lower-50% background defined above, regardless of this option.
