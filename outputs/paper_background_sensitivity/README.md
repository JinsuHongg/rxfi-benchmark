# Reproducing paper-based X-ray backgrounds

## Locked inputs and method

The run uses NOAA/NCEI flare report release `sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv` (SHA-256 `351eda39efea3f5eb4d394b91956d1c08e7f63993c41d09d686e5ad2983cbbea`) and the 43 satellite/year files listed in [`configs/paper_background_sensitivity.inputs.lock.json`](../../configs/paper_background_sensitivity.inputs.lock.json). The lock file itself is pinned in [`configs/paper_background_sensitivity.json`](../../configs/paper_background_sensitivity.json). XRS inputs are science-quality Level 2 `avg1m`, version `v2-2-1`, channel `xrsb_flux`; only `xrsb_flag == 0` measurements are valid. The method and project RXFI definition are described in [`docs/research/paper_based_background_xray_flux_method.md`](../../docs/research/paper_based_background_xray_flux_method.md).

For each event, the calculation uses same-satellite observations from `[start_time - 24 hours, start_time)` in UTC, excludes invalid measurements and other-flare intervals, then computes an arithmetic mean with and without the inclusive cap `xrsb_flux <= flux_at_exact_start_time`. The start value must be an exact valid observation on the event's source satellite. Project RXFI is `(peak_flux - background) / background`; NOAA's event-level `background_irrad` is retained for comparison.

To preserve the existing frozen event table exactly, the replay retains its historical calendar-year interval lookup: it considers flare intervals whose start timestamp falls in one of the UTC calendar years touched by the 24-hour window. A catalog flare that starts in the preceding year and overlaps the window can therefore be omitted at a year boundary. This is a legacy reproducibility limitation, not a requirement stated by the paper.

## Environment and command

Create or update the project environment from the repository's `environment.yml`. In addition to the existing packages, the NetCDF reader uses `xarray==2026.4.0` and `h5netcdf==1.8.1`.

From the repository root, run:

```bash
python scripts/compute_paper_background_sensitivity.py --offline
```

The default catalog is `data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv`; the XRS cache is under `data/raw/noaa_paper_background/xrs_l2/`. The runner accepts the old unversioned cache layout only when its bytes match the lock, and new downloads are stored under a product-version directory. Offline mode fails before publishing outputs if any required file is missing, has a different size or checksum, or cannot be read as the expected NetCDF schema.

An online run can download missing locked files by omitting `--offline`. A custom location for the same catalog bytes can be supplied with `--catalog`; its checksum must still match the configured release. `--cache-dir` changes the cache root, `--output-root` changes the parent for run directories, and `--overwrite` replaces only the directory for the identical fingerprint.

Each successful invocation writes a new fingerprinted folder under `outputs/paper_background_sensitivity/runs/<fingerprint>/` containing `event_backgrounds.csv`, `summary.json`, `input_manifest.json`, and `run_config.json`. Fingerprints depend on the canonical configuration, lock, and input checksums, not retrieval timestamps. The root-level outputs in this directory remain the frozen reference and are not overwritten by the runner.

## Forecast windows

This run produces flare-event backgrounds only. The frozen row-level split files are unavailable in this checkout, so the event estimates have not been aggregated into forecast-window targets. Existing target-generation outputs remain unchanged.
