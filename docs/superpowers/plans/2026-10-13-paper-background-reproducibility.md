# Paper-Based Background Reproducibility Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Keep the current calculated files as a frozen reference while refactoring.

**Goal:** Make the paper-based 24-hour GOES XRS background calculation repeatable from pinned NOAA inputs, explicit method settings, and a reproducible command, without changing the existing event-level results or project targets.

**Architecture:** Move the method and NOAA release identifiers into a versioned JSON configuration. Separate NOAA download/read logic from event calculations, keep the current script as a thin CLI, and write each run to a fingerprinted output directory with input checksums and a machine-readable manifest. Pin `xarray` and `h5netcdf` explicitly and use them for CF time decoding and NetCDF reads.

**Tech Stack:** Python 3.11, standard-library `argparse`/`csv`/`urllib`/`hashlib`, NumPy, pinned `xarray`/`h5netcdf`, pytest for pure-function and small-fixture checks.

**Spec:** `docs/research/paper_based_background_xray_flux_method.md` and the fixed method/results in `docs/research/paper_background_cap_sensitivity.md`.

## Global Constraints

- Pin catalog `sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv` and its SHA-256; do not silently follow a moving “latest” catalog URL.
- Pin XRS L2 `avg1m` version `v2-2-1`, channel `xrsb_flux`, valid quality `xrsb_flag == 0`, and the event's `xrsb_irrad_source` satellite.
- Preserve the 2010–2025 peak-time cohort, exact `(flare_id, time)` deduplication, UTC interval `[start_time - 24 hours, start_time)`, and inclusive mask for other flares' `[start_time, end_time]` intervals.
- Require the exact valid same-satellite XRS sample at `start_time` for the capped result; never interpolate or fall back to another satellite.
- Calculate project RXFI as `(xrsb_irrad - background) / background`; preserve valid negative uncapped values.
- Keep NOAA `background_irrad` as a comparison field, not as the paper-based estimator.
- Never overwrite the current `outputs/paper_background_sensitivity/` reference files or existing derived targets during the refactor.
- Make offline reruns possible from the downloaded local cache and fail clearly when required input files are missing or their pinned checksums differ.
- Preserve the frozen result's historical interval lookup: candidate flare intervals are selected by UTC start year among the calendar years touched by the measurement window. This retains exact prior rows but can omit a prior-year-started flare that overlaps the window across New Year's; record this limitation in the run documentation.

## Review Focus

- A cache file belongs to a specific product version and satellite/year; version changes must not reuse a same-named file silently.
- A download interrupted midway must not appear as a complete cached NetCDF file.
- Missing or invalid start-time observations must make only the capped estimate unavailable; the uncapped estimate should still be computed when its window data are present.
- An event with no samples at or below a valid cap must have a distinct capped-status reason and no fabricated background.
- Rerunning with the same pinned inputs must reproduce the event CSV values and output fingerprint; retrieval timestamps must not affect the fingerprint.
- Events near year boundaries must load both relevant annual XRS files; the historical catalog-mask year lookup is preserved for exact regression reproduction.

---

### Task 1: Pin the method contract and establish regression fixtures

**Files:**
- Create: `configs/paper_background_sensitivity.json`
- Create: `configs/paper_background_sensitivity.inputs.lock.json`
- Create: `tests/test_paper_background_sensitivity.py`
- Reference: `outputs/paper_background_sensitivity/event_backgrounds.csv`
- Reference: `outputs/paper_background_sensitivity/summary.json`

**Interfaces:**
- Config records the exact NOAA catalog URL/name/hash, input lock path, cohort years, XRS product version, fields, validity rule, same-satellite policy, time boundaries, other-flare mask boundaries, cap rule, RXFI formula, and output precision.
- The input lock records the 43 required satellite/year URLs, NOAA product version, byte sizes, and SHA-256 values from the current manifest; it is the expected checksum source for downloads and offline cache validation.
- Config includes an explicit `algorithm_version` and `output_schema_version`; both are fingerprint inputs so a code or schema revision cannot collide with an earlier run.
- Tests will target pure functions introduced in Task 2; begin with test cases for the frozen behavior and expected aggregate counts.

- [x] **Step 1: Add configuration and input lock with the pinned current source release and method values.** Record catalog SHA-256 `351eda39efea3f5eb4d394b91956d1c08e7f63993c41d09d686e5ad2983cbbea`, all 43 XRS file hashes from the current `input_manifest.json`, XRS version `v2-2-1`, 2010–2025, 86,400 seconds, and all method rules from Global Constraints.
- [x] **Step 2: Capture regression expectations from the current outputs.** Pin 41,723 cohort events, 41,723 uncapped estimates, 41,220 capped estimates, 4,842 negative uncapped RXFI values, and the current event CSV row order and numeric columns as comparison evidence.
- [x] **Step 3: Add named behavior cases to the test file.** Cover exact deduplication, UTC/year boundaries, strict pre-start selection, inclusive other-flare mask endpoints, nonzero quality flags, missing/invalid exact start sample, zero retained capped samples, and a valid negative uncapped RXFI.
- [x] **Step 4: Review config and lock against both research documents and the current run manifest.** The checkable result is one method config plus a checksummed NOAA input lock that together capture every choice and input needed to reproduce this run.

### Task 2: Separate calculation rules into tested pure functions

**Files:**
- Create: `scripts/paper_background/__init__.py`
- Create: `scripts/paper_background/calculation.py`
- Modify: `tests/test_paper_background_sensitivity.py`

**Interfaces:**
- `deduplicate_catalog_rows(rows) -> (rows, duplicate_count)` removes only repeated exact `(flare_id, time)` keys.
- `valid_measurement_mask(flux, flags) -> bool_array` implements finite, positive flux and zero quality flag.
- `mask_other_flare_samples(sample_seconds, intervals, target_flare_id) -> bool_array` returns samples inside any other event's inclusive interval; use indexed disjoint interval segments and binary search, preserving the legacy start-year candidate selection above.
- `calculate_event_backgrounds(sample_seconds, flux, flags, start_seconds, start_threshold, peak_flux, flare_intervals, target_flare_id, *, window_seconds=86400, expected_minutes=1440, start_sample_status=None) -> dict` returns both background/RXFI variants, sample counts, coverage, and separate status fields.

- [x] **Step 1: Implement pure function tests for the method boundaries.** Each named case from Task 1 must assert exact retained timestamps/counts and expected numeric formula.
- [x] **Step 2: Implement `deduplicate_catalog_rows` and `valid_measurement_mask`.** Do not discard nonduplicate rows solely because an optional metadata value is empty.
- [x] **Step 3: Implement interval masking with inclusive flare bounds.** Merge overlapping intervals before querying samples; exclude only the target event's own ID.
- [x] **Step 4: Implement paired capped and uncapped arithmetic means and fractional excess.** Reuse the same quality-screened and flare-masked pool; apply only `flux <= start_threshold` to the capped pool.
- [x] **Step 5: Run the focused unit tests.** Run `pytest -q tests/test_paper_background_sensitivity.py`; all boundary cases must pass.

### Task 3: Make NOAA acquisition and NetCDF reads pinned and verifiable

**Files:**
- Create: `scripts/paper_background/noaa.py`
- Modify: `environment.yml` only if the selected reader is not already listed
- Modify: `tests/test_paper_background_sensitivity.py`

**Interfaces:**
- `catalog_url(config) -> str` returns the exact pinned URL.
- `l2_url(satellite, year, version) -> str` returns the correct NCEI path for GOES-13/14/15 or NGDC path for GOES-16+.
- `download_atomic(url, destination, expected_sha256=None, timeout_seconds=60) -> input_record` streams to a temporary file, validates the response and optional checksum, then atomically renames it.
- `read_l2_year(path) -> (seconds, flux, flags, file_metadata)` uses pinned `xarray` with the `h5netcdf` engine and validates required variables and matching lengths.

- [x] **Step 1: Test versioned cache naming and URL construction.** Cache path includes the product version, satellite, and year; tests pin the old/new NOAA host routes.
- [x] **Step 2: Test failed and interrupted downloads with injected `urlopen` responses.** The sandbox prevents local server sockets; cover HTTP errors, HTML, interrupted reads, zero-byte responses, and checksum mismatches, ensuring no final cache file remains and each failure is actionable.
- [x] **Step 3: Pin `xarray` and `h5netcdf` in `environment.yml` and use the explicit `h5netcdf` engine.** The current runtime has these libraries and decodes NOAA CF timestamps; pin their observed compatible versions so the script does not depend on ambient packages.
- [x] **Step 4: Verify NetCDF schema and arrays.** Reject missing `time`, `xrsb_flux`, or `xrsb_flag`, mismatched lengths, and unreadable files; preserve UTC seconds and stable chronological order.
- [x] **Step 5: Run focused NOAA-reader tests.** Use a tiny temporary NetCDF fixture and injected responses; do not contact NOAA in unit tests.

### Task 4: Turn the existing script into a safe reproducible CLI

**Files:**
- Modify: `scripts/compute_paper_background_sensitivity.py`
- Create: `scripts/paper_background/runner.py`
- Modify: `tests/test_paper_background_sensitivity.py`

**Interfaces:**
- CLI accepts `--config`, `--catalog` (optional local pinned copy), `--cache-dir`, `--output-root`, `--offline`, and `--overwrite`.
- Default config is `configs/paper_background_sensitivity.json`; its pinned input lock is `configs/paper_background_sensitivity.inputs.lock.json`; default cache is `data/raw/noaa_paper_background/`.
- Run output is `outputs/paper_background_sensitivity/runs/<fingerprint>/`, where `<fingerprint>` depends only on canonical config, algorithm/output schema versions, and source SHA-256 values, never retrieval time.
- Existing root-level outputs remain the reference dataset and are read-only inputs for regression comparison.

- [x] **Step 1: Add CLI parsing and validate arguments before creating output directories.** A custom local catalog is checksum-verified and its actual path/name appears in the manifest.
- [x] **Step 2: Add input preflight.** Verify config, lock-file hash, catalog and all required satellite/year checksums, and NetCDF schema before event calculation; in offline mode, stop before writing partial outputs if inputs are missing.
- [x] **Step 3: Group catalog events by satellite/year and calculate each background through the pure calculation API.** Retain one event output row for every eligible event with explicit uncapped and capped statuses.
- [x] **Step 4: Create a stable run fingerprint and write outputs atomically.** Write `event_backgrounds.csv`, `summary.json`, `input_manifest.json`, and `run_config.json`; keep retrieval time in the manifest but exclude it from the fingerprint and numeric CSV.
- [x] **Step 5: Protect existing runs.** Refuse to replace a populated fingerprint directory unless `--overwrite` is supplied; never write to or modify project target files.
- [x] **Step 6: Add CLI integration checks.** A small fixture run produces stable CSV bytes across two offline invocations; a missing file fails before publishing output.

### Task 5: Recompute against the frozen reference and document the run procedure

**Files:**
- Create: `outputs/paper_background_sensitivity/README.md`
- Modify: `docs/research/paper_background_cap_sensitivity.md`
- Modify: `docs/research/paper_based_background_xray_flux_method.md`
- Modify: `docs/research/current_state.md`

- [x] **Step 1: Recompute offline from the existing cache into a new fingerprint directory.** The run is under `outputs/paper_background_sensitivity/runs/`; frozen reference files were not changed.
- [x] **Step 2: Compare against the current frozen event output.** The event CSV and summary JSON are byte-identical to the reference, including 41,723 rows and all values/counts.
- [x] **Step 3: Verify provenance.** All 43 locked XRS files are represented with matching size and SHA-256; the fixture test confirms stable fingerprints across offline reruns.
- [x] **Step 4: Document clean-environment setup and one copyable command.** Explain the pinned catalog release, cache location, offline/online modes, output path, checksum behavior, and the intentional absence of forecast-window aggregation while frozen split files are unavailable.
- [x] **Step 5: Update research status and source references.** Keep the NOAA catalog `background_irrad` comparison-only and preserve the existing RXFI convention.

**Checkable result:** a clean environment can recreate the existing event-level results offline from the pinned cache, and a later NOAA release requires an explicit config/version change that yields a separate run fingerprint.

## Source references

- Method: Hong et al., [*Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*](https://par.nsf.gov/servlets/purl/10435332), Section 2.1.
- Pinned catalog directory: [NOAA/NCEI science-quality Flare Report CSV](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/).
- Product guide: [GOES-R XRS L2 Data User's Guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf).
