# Paper-Based 24-Hour Background X-Ray Flux Method

## Purpose and source

This note records the background X-ray flux method described in Hong et al., *Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*. The project intends to use this method to calculate an event-specific background from GOES time-series data, rather than treating NOAA's event-level `background_irrad` field as the background estimator. The paper is the source for the procedure below; implementation details that it does not specify are listed separately.

Source: [NSF Public Access Repository PDF](https://par.nsf.gov/servlets/purl/10435332), Section 2.1, “Relative Increase of Background X-ray Flux.”

## Method described in the paper

The paper uses one-minute averaged GOES X-ray flux measurements. For a flare with start time \(t_0\), it considers the 24-hour period immediately before the flare starts. It removes measurements that meet any of these conditions:

1. The measurement falls between the start and end times of another flare.
2. The measurement is greater than the X-ray flux at the start of the flare of interest.
3. The instrument identifies the measurement as low quality.

The background \(B\) is the arithmetic mean of the remaining valid measurements in that period:

\[
B = \frac{1}{|V|}\sum_{k\in V} X_k,
\]

where \(V\) is the set of one-minute observations that remain after filtering and \(X_k\) is the measured X-ray flux. The paper's example uses the 24 hours before the 2015-09-20 M1.5 flare start (2015-09-19 04:55 to 2015-09-20 04:55); its figure shows the surrounding 48-hour time series for context.

The paper describes its event-level RXFI as peak flux divided by background flux:

\[
\mathrm{RXFI}_{\mathrm{paper}} = \frac{F_{\mathrm{peak}}}{B}.
\]

This is a peak-to-background ratio. A fractional excess above background is instead \((F_{\mathrm{peak}}-B)/B = \mathrm{RXFI}_{\mathrm{paper}}-1\). Keep these two conventions distinct when comparing labels.

## How this differs from the NOAA catalog field

NOAA's flare catalog provides `background_irrad` as an event-level background irradiance recorded at flare start. The paper's method independently estimates background by averaging filtered one-minute measurements over the preceding 24 hours. These values are therefore produced by different procedures and need not agree. For a direct comparison, compute the 24-hour estimate for the same flare event and compare it with that event's NOAA `background_irrad`, while retaining event time, measurement source, and quality information.

## Project interpretation and status

The selected project background estimator is the paper-based filtered 24-hour mean. The NOAA catalog remains useful for identifying flare events, start/end times, and peak measurements, but its `background_irrad` is a comparison value rather than the estimator specified in this note.

As of 2026-10-13, the repository's existing target-generation code still computes RXFI using the NOAA catalog's `background_irrad`. This note documents the selected methodology; it does not change code or regenerate derived targets. Future implementation and comparison should record the paper-based background separately from the NOAA field. If the project retains its existing fractional-excess convention, calculate RXFI as \((F_{\mathrm{peak}}-B)/B\), and state that this is one less than the paper's peak-to-background ratio.

The event-level implementation and pinned NOAA inputs are reproducible with [`scripts/compute_paper_background_sensitivity.py`](../../scripts/compute_paper_background_sensitivity.py), [`configs/paper_background_sensitivity.json`](../../configs/paper_background_sensitivity.json), and its checksummed input lock. Run instructions and cache behavior are in [`outputs/paper_background_sensitivity/README.md`](../../outputs/paper_background_sensitivity/README.md). The runner writes versioned, fingerprinted results beneath `outputs/paper_background_sensitivity/runs/` and leaves the root-level reference files untouched.

The frozen result preserves a historical calendar-year lookup detail: candidate other-flare intervals are selected by their UTC start year within the year or years touched by the 24-hour measurement window. Consequently, an interval starting in the preceding year but overlapping a window that begins after New Year's can be omitted. This edge case is retained for exact result reproduction and is not a rule stated by the paper; it should be changed explicitly in a future method-version update.

## Details the paper leaves unspecified

Section 2.1 does not fully specify the following implementation choices. They should be fixed and documented before results are treated as reproducible:

- The exact GOES product, channel, satellite-selection rule, and calibration/version used for the one-minute series.
- The flare catalog or event-identification source used to mask other flares, and whether the start/end boundaries are inclusive.
- The exact quality-flag values considered low quality.
- Handling of missing minutes, gaps, duplicate timestamps, or no valid samples after filtering.
- Whether observations are equally weighted when averaging (the text says to average valid measurements but gives no alternative weighting scheme).
- The exact timestamp convention and boundary inclusion for the 24-hour interval.

These are implementation decisions, not additional rules asserted by the paper.
