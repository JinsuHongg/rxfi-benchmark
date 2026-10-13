# Background calculations

Calculates event-level GOES XRS-B background irradiance from the preceding 24 hours using the paper-based method. The runner uses checksum-pinned NOAA flare catalog and Level 2 XRS inputs and reports capped and uncapped estimates.

- `compute_paper_background_sensitivity.py`: command-line entry point for a reproducible calculation. From the repository root: `python scripts/background/compute_paper_background_sensitivity.py --offline`.
- [`paper_background/`](paper_background/README.md): input retrieval, quality handling, event-background calculation, and run orchestration.

See the [background method note](../../docs/research/paper_based_background_xray_flux_method.md) and [run guide](../../outputs/paper_background_sensitivity/README.md) for sources, assumptions, and options.
