# Background calculations

Calculates event-level GOES XRS-B background irradiance from the preceding 24 hours using the paper-based method or the project's lower-50% adaptation. The runner uses checksum-pinned NOAA flare catalog and Level 2 XRS inputs; the paper-method configuration reports capped and uncapped estimates.

For the project-adapted lower-50% estimator and its RXFI outputs, use `configs/background_lower50.json`. This keeps only the lowest half of eligible pre-event flux samples after quality filtering and other-flare masking. Run it in the project environment from the repository root:

```bash
conda run -n rxfi-benchmark python scripts/background/compute_paper_background_sensitivity.py --config configs/background_lower50.json --offline
```

The method, output columns, and interpretation are documented in [`docs/research/lower50_background_rxfi.md`](../../docs/research/lower50_background_rxfi.md). The result is written under `outputs/background_lower50/runs/`.

- `compute_paper_background_sensitivity.py`: command-line entry point for a reproducible calculation. From the repository root: `python scripts/background/compute_paper_background_sensitivity.py --offline`.
- `plot_capped_background_example.py`: recreates the June 14, 2010 example figure from the pinned GOES XRS and flare catalog files. Run `python scripts/background/plot_capped_background_example.py`; it writes PNG and SVG to `outputs/paper_background_sensitivity/figures/` and verifies the displayed counts and uncapped mean against the saved event result.
- `plot_missing_x_class_backgrounds.py`: plots all X-class flares whose capped background is unavailable, with the previous 24 hours, flare start cap, and flare-time XRS-B curve. Run `python scripts/background/plot_missing_x_class_backgrounds.py`; it writes PNG and SVG to `outputs/paper_background_sensitivity/figures/` and checks plotted candidate counts against the saved event results.
- `plot_x_class_background_calculation.py`: explains the selected lower-50% background for the 2011-09-22 X2.1 flare in one GOES-15 XRS-B time-series plot. It highlights the lower-half samples and calculated background, marks other flare intervals excluded from the background with orange dots and matching shading, and shades the target flare interval; checks the calculation against the saved event table; and writes PNG/SVG to `outputs/background_lower50/figures/`. Run from the repository root in the project environment:

  ```bash
  conda run -n rxfi-benchmark python scripts/background/plot_x_class_background_calculation.py
  ```

  The figure uses the checksum-pinned NOAA flare catalog and GOES-15 Level 2 `avg1m` data. It shows only the project's lower-50% background calculation; NOAA catalog `background_irrad` is not plotted.
- `plot_x_class_prediction_window.py`: plots the 24-hour interval from 2011-09-21 12:00 to 2011-09-22 12:00 UTC on GOES-15 XRS-B Level 2 `avg1m` data and marks each NOAA catalog flare peak in the interval with a red star and its flare-class label. It writes PNG/SVG to `outputs/background_lower50/figures/`. Run from the repository root in the project environment:

  ```bash
  conda run -n rxfi-benchmark python scripts/background/plot_x_class_prediction_window.py
  ```

  The plotted curve uses valid, positive XRS-B samples with `xrsb_flag == 0`; star locations use the catalog peak time and `xrsb_irrad`.
- [`paper_background/`](paper_background/README.md): input retrieval, quality handling, event-background calculation, and run orchestration.

See the [background method note](../../docs/research/paper_based_background_xray_flux_method.md) and [run guide](../../outputs/paper_background_sensitivity/README.md) for sources, assumptions, and options.
