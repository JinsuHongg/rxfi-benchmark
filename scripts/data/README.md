# Data preparation

Scripts for constructing or inspecting NOAA-derived labels and XRS-based no-flare targets.

- `generate_24h_targets.py`: builds four 24-hour peak-flux/RXFI targets from frozen splits, the NOAA flare report, and checksum-pinned science-quality XRS-B measurements. For FQ windows it finds the in-window XRS maximum and computes its lower-50% background over the 24 hours ending five minutes before that maximum; max and cumulative targets are equal.
- `inspect_noaa_xrs_flare_report.py`: inspects flare-report fields and the historical event-level NOAA `background_irrad`/RXFI values used by the existing frozen target files. The selected lower-50% background calculation is in [`../background/README.md`](../background/README.md).
- `refine_no_flare_max_peak_flux.py`: fills no-flare-window maximum flux from quality-filtered GOES XRS 1-minute data and writes coverage/QC outputs.

Run from the repository root; see [target generation notes](../../docs/research/24h_target_generation.md) for input and output contracts.
