# NOAA XRS Flare Catalog and Initial RXFI Definition

## 1. Purpose

This note documents the NOAA/NCEI **GOES Flare Report** selected for Project 1 as the canonical source of flare peak and background X-ray irradiances. It freezes the initial Relative X-ray Flux Increase (RXFI) construction only; it does not specify forecasting models, data splits, or later thesis projects.

## 2. NOAA data source

- Product: NOAA/NCEI Level-2 GOES Flare Report (also called the **XRS Flare Report**; the underlying XRS Flare Summary product is `flsum`).
- Official data directory: <https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/>.
- CSV directory: <https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/>.
- Product documentation: <https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf>.
- CSV variable metadata: <https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/sci_xrsf-l2-flrpt_geo_metadata.json>.
- XRS L2 algorithm documentation: <https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf>.

The inspection below used the official 2024 yearly file, `sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv`, downloaded temporarily rather than committed. The catalog is updated; reproducible analysis should record the exact file name, version, retrieval date, and checksum.

## 3. Verified catalog properties

| Property | Verified NOAA statement |
| --- | --- |
| Product scope | Science-quality composite, event-based flare report derived primarily from GOES XRS L2 science flare summaries, with SWPC reports supplying supplemental event/location information. |
| Current coverage | The documentation describes GOES-08 through GOES-19 from 1995 onward; the current metadata inspected reports `19950103` through `20260913`. Coverage is a release-dependent property. |
| One record per flare | Yes. The report is an event table indexed by flare peak time and has one entry per flare event. |
| Satellite selection | The designated primary satellite is used when available; otherwise secondary or tertiary data are used. When two satellites observe events within 4 minutes and their flare irradiances agree within +/-15%, the primary record is used. Otherwise, two separate flare events are reported. |
| XRS channel | XRS-B, soft X-rays over 0.1-0.8 nm. |
| Sampling | Reported irradiances derive from one-minute XRS-B averages; CSV times have one-minute cadence. |
| Irradiance units | `xrsb_irrad` and `background_irrad` are W/m2. The documentation uses “irradiance”; historical variable names elsewhere may say “flux.” |
| Peak time | `time`: flare peak time / record start time for the flare peak. |
| Peak irradiance | `xrsb_irrad`: one-minute averaged irradiance for XRS-B, at the flare peak. |
| Start time | `start_time`: flare event start time (`EVENT_START`). The XRS algorithm detects a start from good one-minute data and sets a pre-flare background during that process. |
| End time | `end_time`: flare event end time (`EVENT_END`), when the background-subtracted irradiance has declined to one half of the background-subtracted peak; this is not simply the raw irradiance falling to half its peak. |
| Background irradiance | `background_irrad`: background irradiance recorded at flare start. The underlying detection algorithm calls it the pre-flare background; for ordinary starts it is the minimum of a fitted exponential rise, and for a new flare during a declining event it is the minimum non-smoothed flux since the previous peak. |

The last row is important: this is an event-specific algorithmic pre-flare baseline, not the separate daily-background product. NOAA cautions that this estimate can be wrong by roughly +/-20% when pre-flare data are noisy or the flare rises to peak in only several minutes.

### Assumptions and unresolved interpretation

The exact algorithm description comes from the XRS L2 guide and describes the `flsum` input product used by the report. It is reasonable—but not separately stated in the report ReadMe—to treat the report field as that flsum baseline after composite record selection. The available NOAA metadata does not explicitly say whether all historical satellite records use precisely identical algorithm parameterization or whether later catalog revisions can revise individual event values. Those should remain documented limitations rather than assumptions hidden in downstream analyses.

## 4. Relevant CSV schema

The verified CSV header is:

```text
time,start_time,end_time,flare_id,xrsb_irrad,flare_class,xrsb_irrad_source,background_irrad,...,sequential_flare_num,...,peak_saturated,...
```

| Column | Meaning | Unit | Needed for RXFI |
| --- | --- | --- | --- |
| `time` | flare peak time | UTC datetime | Yes, event identity/index |
| `start_time` | flare start time | UTC datetime | Yes, provenance/temporal analysis |
| `end_time` | flare end time | UTC datetime | Optional for RXFI calculation; useful for event QA |
| `flare_id` | identifier based on flare start (`YYYYMMDDHHMM`) | identifier | Recommended for deduplication checks |
| `xrsb_irrad` | one-minute XRS-B peak irradiance | W/m2 | **Yes: F_peak** |
| `flare_class` | class recorded at peak | class label | Recommended QA/context |
| `xrsb_irrad_source` | GOES satellite that measured irradiance | GOES-1..19 label | Recommended provenance/transition QA |
| `background_irrad` | background irradiance recorded at start | W/m2 | **Yes: F_background** |
| `sequential_flare_num` | position in a flare sequence | 1 | Recommended; indicates potentially overlapping events |
| `peak_saturated` | peak saturation indicator | 0/1 | Recommended QA; 1 is saturated |

`integrated_irrad_peak` and `integrated_irrad_end` have J/m2 and are not inputs to the initial RXFI. The event irradiance source is `xrsb_irrad_source`, not a generic GOES field. No separate report-level quality bitmask is present in the inspected schema; `peak_saturated` is the directly relevant reported quality flag. Blank CSV cells represent nonexistent data.

## 5. Satellite selection behavior

NOAA's composite logic is the canonical deduplication/selection policy. Analyses must retain `xrsb_irrad_source` because selected satellite identity can change across the time series, especially at operational-primary transitions or when primary data are unavailable. The selection is event-level: the same flare report event supplies the peak and background fields from the selected source record. It should not be reconstructed by joining satellite-level products unless a later study explicitly requires that different product.

## 6. Proposed frozen initial RXFI definition

For an event with valid positive values:

\[
\operatorname{RXFI} = \frac{F_{\mathrm{peak}} - F_{\mathrm{background}}}{F_{\mathrm{background}}},
\]

where `F_peak = xrsb_irrad` and `F_background = background_irrad`, both in W/m2 from the same selected NOAA flare-report event.

This is dimensionless and is mathematically appropriate as a relative increase above the NOAA event-specific baseline. It is equivalent to `F_peak / F_background - 1`, so it is interpretable as fractional excess (for example, 4 means peak is five times the baseline). It is the frozen initial Project-1 definition, conditional on the eligibility rules below.

| Alternative | Relationship | Reason not selected as canonical initial definition |
| --- | --- | --- |
| `F_peak / F_background` | `RXFI + 1` | Dimensionless, but its value includes the baseline ratio rather than expressing an increase above baseline. |
| `log(F_peak / F_background)` | monotonic transform of ratio | Dimensionless and compresses extremes, but changes the scale and requires strictly positive values; retain only as a possible sensitivity transform. |
| `F_peak - F_background` | numerator of RXFI | Units are W/m2, so it measures absolute excess rather than relative increase. |

## 7. Edge cases and recommended handling

These are recommended analysis rules, not NOAA filtering rules unless stated as verified.

| Case | Verified NOAA behavior | Recommended RXFI handling |
| --- | --- | --- |
| Missing background | Nonexistent values are blank in CSV. | Mark RXFI missing; exclude from primary numeric summaries; report count. |
| Background equal to zero | Metadata valid minimum is about `1e-9 W/m2`; zero is not a valid reported value. | Treat as invalid, never divide; report count. |
| Missing/nonpositive/nonfinite peak | Nonexistent values are blank; metadata gives a positive valid range. | Mark RXFI missing; exclude and report. |
| Duplicate records | NOAA applies deduplication before writing; unmatched multi-satellite observations outside the 4-minute/+/-15% match criteria can intentionally appear as separate flare events. | Do not collapse rows by time alone. Audit exact duplicate `(flare_id, time)` keys; investigate rather than automatically dropping any found. |
| Quality/status flags | `peak_saturated` is a 0/1 field; saturation is documented as rare and historically limited to GOES-1 through GOES-12. | Retain and report saturated events. Exclude from a primary unsaturated analysis, with a sensitivity analysis if their treatment matters. |
| Satellite transitions | Primary designation changes over time and fallback satellites are possible. | Preserve `xrsb_irrad_source`; stratify or sensitivity-check source/transition periods before making cross-era claims. |
| Source changes across time | Catalog source is selected per event; no single source is guaranteed for a time interval when primary data are unavailable. | Treat source as an event-level provenance variable, not a constant era label. |
| Very small backgrounds | The documented minimum is about `1e-9 W/m2`; ratios can still be large. | Do not silently cap values. Report high quantiles, inspect extremes, and pre-register any later robust-transform/winsorization sensitivity analysis. |
| Sequential/overlapping flares | `sequential_flare_num > 1` marks later flares in a sequence; NOAA notes overlapping flares complicate background estimation in related XRS processing. | Retain for the initial catalog, flag for descriptive and sensitivity analyses rather than removing by default. |

## 8. Reproducibility check

Run:

```bash
python3 scripts/inspect_noaa_xrs_flare_report.py /path/to/sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv
```

With no argument, the script downloads its default official 2024 yearly CSV. It prints the RXFI-related schema, counts valid and invalid events, exact duplicate `(flare_id, time)` keys, saturated events, and distribution summaries/quantiles using the frozen formula. It intentionally has no ML functionality and uses only the Python standard library.

## 9. Open questions

1. Confirm with NOAA/NCEI whether the report's `background_irrad` is always copied without transformation from the selected satellite's flsum record across GOES-08 through GOES-19.
2. Pin the catalog release (and checksum) for the thesis dataset, because yearly and mission-length files can be revised.
3. Establish, before inferential modeling, whether source satellite, saturated peaks, and sequential flares require pre-specified sensitivity strata.
4. Confirm the desired final temporal cutoff; the live catalog coverage advances after this verification.

## 10. Proposal-ready methodology wording

> Project 1 will use NOAA/NCEI's science-quality Level-2 GOES Flare Report as the canonical event catalog for GOES XRS flare irradiance and timing. The composite report supplies one selected satellite record per flare event, prioritizing the operational primary GOES observation when matching observations satisfy NOAA's event-matching criteria. For each event, the relative X-ray flux increase will be defined from the one-minute XRS-B (0.1-0.8 nm) peak irradiance and NOAA-reported background irradiance recorded at flare start as \(\mathrm{RXFI}=(F_{\mathrm{peak}}-F_{\mathrm{background}})/F_{\mathrm{background}}\). Events lacking valid positive peak or background irradiance will be reported and excluded from calculations requiring RXFI; satellite source, saturation, and flare-sequence fields will be retained for quality control and sensitivity analyses.
