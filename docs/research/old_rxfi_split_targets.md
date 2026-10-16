# Supplementary maximum old-RXFI split target

## Purpose and scope

This document defines a supplementary catalog-based target for comparing the
historical RXFI ratio with the four XRS-only targets. It does not change the
XRS-only target definitions in
[`xrs_only_target_definitions.md`](xrs_only_target_definitions.md).
The completed comparison with `max_background_deviation` is documented in
[`old_rxfi_vs_background_deviation.md`](old_rxfi_vs_background_deviation.md).

The event-level values come from the copied NOAA catalog
`data/derived/noaa_flare_catalog_old_background_rxfi.csv`. The source catalog
is preserved. Its generation script and background definition are described
in [`../../scripts/background/README.md`](../../scripts/background/README.md)
and [`noaa_xrs_catalog_and_rxfi_definition.md`](noaa_xrs_catalog_and_rxfi_definition.md).

## Event-level old RXFI

For a NOAA catalog event with start time `t_start` and peak irradiance
`F_peak`, the old background `B_old` is the arithmetic mean of eligible XRS-B
samples from the event's catalog-selected GOES satellite in
`[t_start - 24 h, t_start)`, after:

- excluding samples in other catalog flare intervals;
- retaining finite, positive samples with `xrsb_flag == 0`;
- retaining only values no greater than the XRS-B sample at the exact catalog
  start time on the same satellite.

The event-level ratio is

\[
\mathrm{old\_rxfi} = \frac{F_{\mathrm{peak}}}{B_{\mathrm{old}}}.
\]

It is a dimensionless peak-to-background ratio. It is distinct from the signed
`max_background_deviation`, which is an absolute irradiance difference from a
forecast-start background.

## Split-window aggregation

For a split row with timestamp `t`, event membership is determined by the
catalog **flare start time**, not its peak time. The prediction window is the
half-open interval `[t, t + 24 h)`. For eligible events `E_t` whose
`start_time` lies in that window:

\[
\texttt{max\_old\_rxfi} = \max_{i \in E_t} \mathrm{old\_rxfi}_i.
\]

An event is eligible when its catalog-copy status is `ok` and its `old_rxfi`
is finite. Exact duplicate events keyed by `(flare_id, time)` are counted
once; conflicting duplicate values are rejected. If multiple events tie for
the maximum, the earliest `start_time`, then lexical `flare_id`, identifies
`max_old_rxfi_flare_id`.

If the window has no catalog events, or has events but none with a valid
`old_rxfi`, `max_old_rxfi` is missing. It is never imputed as zero. The output
also records the total catalog-event count, valid-RXFI event count, and a
status distinguishing these cases.

## Inputs, outputs, and split roles

The generator reads the existing `data/derived/{split}_targets.csv` files and
the enriched NOAA catalog copy. It does not modify those inputs or the frozen
SuryaBench split membership. Augmented copies are written to
`data/derived/old_rxfi_targets/{split}_targets.csv`; each retains its original
columns, row order, and timestamps and appends `max_old_rxfi` plus audit fields.
The directory also contains per-split QC and a metadata file with source
checksums and the aggregation definition. These derived products are local and
gitignored.

The same temporal split roles are retained: train for fitting, validation for
model selection, `leaky_validation` for diagnostic use, and test for final
evaluation. Test labels must not inform model, target, or procedure choices.
Because forecast windows overlap in time, one event can contribute to more
than one split row; the published split membership itself is not changed.

## Reproduction

Run from the repository root with the project environment after generating the
event-level catalog copy and the four XRS-only target files:

```bash
conda run -n rxfi-benchmark python scripts/data/add_old_max_rxfi_targets.py
```

Pass `--catalog`, `--input-dir`, or `--output-dir` to use alternate local
inputs or destinations. Use `--overwrite` to regenerate existing output
files. No model training is performed by this script.
