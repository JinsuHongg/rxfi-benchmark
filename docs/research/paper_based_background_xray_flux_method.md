# Paper-Based 24-Hour Background X-Ray Flux Method

## Purpose and source

This note records the background X-ray flux method described in Hong et al., *Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*. It is the source-method reference for the project, but the project's selected estimator is now a lower-50% adaptation, documented separately in [lower50_background_rxfi.md](lower50_background_rxfi.md). NOAA's event-level `background_irrad` remains a comparison value for the new event-level analysis. The paper is the source for the procedure below; implementation details that it does not specify are listed separately.

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

The paper's method defines a full arithmetic mean after its quality, flare-interval, and start-time upper-cap filters. The project selected a different aggregation: the mean of the lowest half of eligible samples after quality and other-flare interval filtering, without the upper cap. This is a project adaptation, not a procedure specified by the paper; its exact definition and event-level results are in [lower50_background_rxfi.md](lower50_background_rxfi.md).

The existing frozen forecast-window targets still compute RXFI from NOAA catalog `background_irrad`; they have not been regenerated using the selected lower-50% estimator. The new event-level calculation keeps `background_irrad` for comparison. Project RXFI uses \((F_{\mathrm{peak}}-B)/B\), one less than the paper's peak-to-background ratio \(F_{\mathrm{peak}}/B\).

The paper-method capped/uncapped comparison is reproducible with [`scripts/background/compute_paper_background_sensitivity.py`](../../scripts/background/compute_paper_background_sensitivity.py), [`configs/paper_background_sensitivity.json`](../../configs/paper_background_sensitivity.json), and its checksummed input lock. Run instructions and cache behavior are in [`outputs/paper_background_sensitivity/README.md`](../../outputs/paper_background_sensitivity/README.md). The selected lower-50% estimator uses the same runner with [`configs/background_lower50.json`](../../configs/background_lower50.json) and writes fingerprinted results beneath `outputs/background_lower50/runs/`.

Both runs preserve a historical calendar-year lookup detail: candidate other-flare intervals are selected by their UTC start year within the year or years touched by the 24-hour measurement window. Consequently, an interval starting in the preceding year but overlapping a window that begins after New Year's can be omitted. This edge case is retained for exact result reproduction and is not a rule stated by the paper; it should be changed explicitly in a future method-version update.

## Details the paper leaves unspecified

Section 2.1 does not fully specify the following implementation choices. They should be fixed and documented before results are treated as reproducible:

- The exact GOES product, channel, satellite-selection rule, and calibration/version used for the one-minute series.
- The flare catalog or event-identification source used to mask other flares, and whether the start/end boundaries are inclusive.
- The exact quality-flag values considered low quality.
- Handling of missing minutes, gaps, duplicate timestamps, or no valid samples after filtering.
- Whether observations are equally weighted when averaging (the text says to average valid measurements but gives no alternative weighting scheme).
- The exact timestamp convention and boundary inclusion for the 24-hour interval.

These are implementation decisions, not additional rules asserted by the paper.
