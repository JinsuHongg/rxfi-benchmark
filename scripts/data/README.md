# Data preparation

Scripts for constructing or inspecting NOAA-derived labels and XRS-based no-flare targets.

- `generate_24h_targets.py`: builds four 24-hour peak-flux/RXFI targets from frozen splits and a NOAA flare-report CSV.
- `inspect_noaa_xrs_flare_report.py`: inspects flare-report fields and event-level NOAA background/RXFI values.
- `refine_no_flare_max_peak_flux.py`: fills no-flare-window maximum flux from quality-filtered GOES XRS 1-minute data and writes coverage/QC outputs.

Run from the repository root; see [target generation notes](../../docs/research/24h_target_generation.md) for input and output contracts.
