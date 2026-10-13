"""Reproducible, checksum-locked orchestration for paper-style backgrounds."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .calculation import calculate_event_backgrounds, deduplicate_catalog_rows, index_flare_intervals
from .noaa import (
    InputDownloadError,
    cache_path,
    catalog_url,
    download_atomic,
    find_cached_input,
    read_l2_year,
    sha256_file,
    verify_input_file,
)

UTC = timezone.utc
EVENT_FIELDS = [
    "time", "start_time", "end_time", "flare_id", "flare_class", "satellite",
    "xrsb_irrad_source", "peak_flux_W_m2", "noaa_background_irrad_W_m2",
    "start_threshold_flux_W_m2", "start_sample_status", "expected_minutes",
    "observed_minutes", "valid_quality_minutes", "other_flare_masked_minutes",
    "uncapped_retained_minutes", "capped_removed_by_threshold_minutes",
    "capped_retained_minutes", "uncapped_coverage_fraction", "capped_coverage_fraction",
    "background_uncapped_W_m2", "background_capped_W_m2",
    "rxfi_uncapped_fractional_excess", "rxfi_capped_fractional_excess",
    "uncapped_status", "capped_status",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        except ValueError:
            continue
    return None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, precision: str) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return format(number, precision) if math.isfinite(number) else ""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"catalog has no CSV header: {path}")
        needed = {"time", "start_time", "end_time", "flare_id", "xrsb_irrad", "xrsb_irrad_source"}
        missing = sorted(needed.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"catalog is missing required column(s): {', '.join(missing)}")
        return list(reader)


def run_calculation(
    config_path: Path,
    *,
    catalog_path: Path | None = None,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    offline: bool = True,
    overwrite: bool = False,
) -> Path:
    """Preflight pinned inputs, calculate events, and atomically publish a run."""
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    base = config_path.parent.parent
    lock_reference = Path(config["input_lock"])
    lock_path = (config_path.parent.parent / lock_reference).resolve()
    if not lock_path.is_file():
        lock_path = (config_path.parent / lock_reference).resolve()
    lock_bytes = lock_path.read_bytes()
    expected_lock_sha = config.get("input_lock_sha256")
    actual_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    if expected_lock_sha and actual_lock_sha.lower() != expected_lock_sha.lower():
        raise InputDownloadError(
            f"input lock SHA-256 mismatch for {lock_path}: expected {expected_lock_sha}, received {actual_lock_sha}"
        )
    lock = json.loads(lock_bytes)
    catalog_config, xrs_config = config["catalog"], config["xrs_l2"]
    lock_catalog = lock["catalog"]
    if (lock_catalog["sha256"].lower() != catalog_config["sha256"].lower()
            or lock_catalog["filename"] != catalog_config["filename"]):
        raise InputDownloadError("catalog configuration and input lock disagree on filename or SHA-256")
    catalog_source = Path(catalog_path).resolve() if catalog_path else (
        base / "data/raw/noaa_paper_background" / catalog_config["filename"]
    )
    catalog_downloaded = False
    if not catalog_source.is_file():
        if offline:
            raise FileNotFoundError(f"pinned NOAA catalog is not available offline: {catalog_source}")
        record = download_atomic(
            catalog_url(config), catalog_source, expected_sha256=catalog_config["sha256"],
            expected_size=lock_catalog.get("size_bytes"),
        )
        catalog_downloaded = True
        catalog_retrieved_at = record["retrieved_at_utc"]
    else:
        catalog_retrieved_at = None
    catalog_record = verify_input_file(
        catalog_source, catalog_config["sha256"], lock_catalog.get("size_bytes")
    )
    if catalog_source.name != catalog_config["filename"]:
        # A caller may provide an alternate path, but it still represents the pinned release.
        catalog_record["provided_filename"] = catalog_source.name
    source_rows = _catalog_rows(catalog_source)
    unique_rows, duplicate_count = deduplicate_catalog_rows(source_rows)

    peak_min = _parse_time(catalog_config["peak_time_start_utc"])
    peak_max = _parse_time(catalog_config["peak_time_end_exclusive_utc"])
    if peak_min is None or peak_max is None:
        raise ValueError("invalid peak-time cohort boundaries in config")
    cohort: list[dict[str, Any]] = []
    counts = Counter()
    intervals: list[tuple[int, int, str]] = []
    for row in unique_rows:
        start, end = _parse_time(row.get("start_time", "")), _parse_time(row.get("end_time", ""))
        if start is not None and end is not None and end >= start:
            intervals.append((int(start.timestamp()), int(end.timestamp()), row.get("flare_id", "")))
        peak_time = _parse_time(row.get("time", ""))
        if peak_time is None or start is None:
            counts["events_missing_time_or_invalid_timing"] += 1
            continue
        if not (peak_min <= peak_time < peak_max):
            continue
        source = row.get(xrs_config["satellite_field"], "")
        if not source.startswith("GOES-"):
            counts["events_missing_source"] += 1
            continue
        peak = _positive_float(row.get("xrsb_irrad", ""))
        if peak is None:
            counts["events_missing_valid_peak"] += 1
            continue
        try:
            satellite = int(source.split("-", 1)[1])
        except ValueError:
            counts["events_missing_source"] += 1
            continue
        cohort.append({**row, "satellite": satellite, "t0": int(start.timestamp()),
                       "peak_time": peak_time, "peak_flux": peak})
    cohort.sort(key=lambda event: (event.get("time", ""), event.get("flare_id", "")))
    needed = sorted({(event["satellite"], year)
                     for event in cohort
                     for year in range(
                         datetime.fromtimestamp(event["t0"] - int(config["method"]["window_seconds"]), UTC).year,
                         datetime.fromtimestamp(event["t0"], UTC).year + 1,
                     )})

    cache_root = Path(cache_root or (base / "data/raw/noaa_paper_background")).resolve()
    lock_files = {(int(item["satellite"]), int(item["year"])): item for item in lock["xrs_l2"]["files"]}
    missing_lock = [key for key in needed if key not in lock_files]
    if missing_lock:
        raise InputDownloadError(f"input lock lacks required satellite/year entries: {missing_lock[:10]}")
    series: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    manifest_files = []
    retrieval_times: dict[str, str | None] = {}
    # Validate and read every required file before creating an output directory.
    for satellite, year in needed:
        locked = lock_files[(satellite, year)]
        path = cache_path(cache_root, satellite, year, xrs_config["version"])
        try:
            path, record = find_cached_input(
                cache_root, satellite, year, xrs_config["version"],
                expected_sha256=locked["sha256"], expected_size=locked["size_bytes"],
            )
            downloaded = False
        except FileNotFoundError:
            if offline:
                raise
            record = download_atomic(
                locked["url"], path, expected_sha256=locked["sha256"], expected_size=locked["size_bytes"]
            )
            downloaded = True
        seconds, flux, flags, metadata = read_l2_year(path)
        series[(satellite, year)] = (seconds, flux, flags)
        retrieval_times[f"GOES-{satellite}/{year}"] = record.get("retrieved_at_utc")
        manifest_files.append({
            **locked, "path": str(path), "legacy_layout": bool(record.get("legacy_layout", False)),
            "downloaded_this_run": downloaded, "retrieved_at_utc": record.get("retrieved_at_utc"),
            "sample_count": metadata["sample_count"],
        })

    # Capture a provided local catalog's checksum and parse all files before output publication.
    manifest = {
        "manifest_schema_version": 1,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "catalog": {**catalog_record, "url": catalog_url(config),
                    "downloaded_this_run": catalog_downloaded,
                    "retrieved_at_utc": catalog_retrieved_at},
        "xrs_l2_product": xrs_config["product"], "xrs_l2_version": xrs_config["version"],
        "files": manifest_files,
    }
    interval_indexes_by_start_years: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    precision = config.get("output", {}).get("float_format", ".12g")
    expected_minutes = int(config["method"].get("window_seconds", 86400) // 60)
    event_rows: list[dict[str, Any]] = []
    for event in cohort:
        sat, t0 = event["satellite"], event["t0"]
        lower = t0 - int(config["method"]["window_seconds"])
        years = list(range(datetime.fromtimestamp(lower, UTC).year, datetime.fromtimestamp(t0, UTC).year + 1))
        year_key = tuple(years)
        if year_key not in interval_indexes_by_start_years:
            included_years = set(years)
            intervals_for_window = [
                interval for interval in intervals
                if datetime.fromtimestamp(interval[0], UTC).year in included_years
            ]
            interval_indexes_by_start_years[year_key] = index_flare_intervals(intervals_for_window)
        parts = []
        for year in years:
            seconds, flux, flags = series[(sat, year)]
            left, right = np.searchsorted(seconds, lower, side="left"), np.searchsorted(seconds, t0, side="left")
            parts.append((seconds[left:right], flux[left:right], flags[left:right]))
        seconds = np.concatenate([part[0] for part in parts]) if parts else np.empty(0, dtype=np.int64)
        flux = np.concatenate([part[1] for part in parts]) if parts else np.empty(0, dtype=float)
        flags = np.concatenate([part[2] for part in parts]) if parts else np.empty(0, dtype=np.int16)
        threshold = None
        sample_status = "missing_exact_sample"
        for year in years:
            samples, values, quality = series[(sat, year)]
            pos = int(np.searchsorted(samples, t0, side="left"))
            if pos < len(samples) and int(samples[pos]) == t0:
                if np.isfinite(values[pos]) and values[pos] > 0 and quality[pos] == 0:
                    threshold, sample_status = float(values[pos]), "valid"
                else:
                    sample_status = "invalid_quality_or_flux"
                break
        calc = calculate_event_backgrounds(
            seconds, flux, flags, t0, threshold, event["peak_flux"], (), str(event.get("flare_id", "")),
            window_seconds=int(config["method"]["window_seconds"]), expected_minutes=expected_minutes,
            start_sample_status=sample_status,
            indexed_flare_intervals=interval_indexes_by_start_years[year_key],
        )
        event_rows.append({
            "time": event.get("time", ""), "start_time": event.get("start_time", ""),
            "end_time": event.get("end_time", ""), "flare_id": event.get("flare_id", ""),
            "flare_class": event.get("flare_class", ""), "satellite": f"GOES-{sat}",
            "xrsb_irrad_source": event.get("xrsb_irrad_source", ""),
            "peak_flux_W_m2": _fmt(event["peak_flux"], precision),
            "noaa_background_irrad_W_m2": event.get("background_irrad", ""),
            "start_threshold_flux_W_m2": _fmt(threshold, precision), "start_sample_status": sample_status,
            "expected_minutes": expected_minutes, "observed_minutes": calc["candidate_count"],
            "valid_quality_minutes": calc["valid_quality_count"],
            "other_flare_masked_minutes": calc["other_flare_masked_count"],
            "uncapped_retained_minutes": calc["uncapped_retained_count"],
            "capped_removed_by_threshold_minutes": calc["capped_removed_by_threshold_count"] if threshold is not None else "",
            "capped_retained_minutes": calc["capped_retained_count"] if threshold is not None else "",
            "uncapped_coverage_fraction": _fmt(calc["uncapped_coverage_fraction"], precision),
            "capped_coverage_fraction": _fmt(calc["capped_coverage_fraction"], precision),
            "background_uncapped_W_m2": _fmt(calc["background_uncapped"], precision),
            "background_capped_W_m2": _fmt(calc["background_capped"], precision),
            "rxfi_uncapped_fractional_excess": _fmt(calc["rxfi_uncapped"], precision),
            "rxfi_capped_fractional_excess": _fmt(calc["rxfi_capped"], precision),
            "uncapped_status": calc["uncapped_status"], "capped_status": calc["capped_status"],
        })

    status_counts = Counter(row["uncapped_status"] for row in event_rows)
    summary = {
        "catalog_file": catalog_config["filename"], "catalog_sha256": catalog_record["sha256"],
        "l2_product": lock["xrs_l2"]["product"], "l2_version": xrs_config["version"],
        "cohort_peak_time_range_utc": [peak_min.strftime("%Y-%m-%d"), (peak_max - timedelta(days=1)).strftime("%Y-%m-%d")],
        "catalog_rows": len(source_rows), "exact_duplicate_rows_removed": duplicate_count,
        "deduplicated_catalog_rows": len(unique_rows),
        "events_missing_time_or_invalid_timing": counts["events_missing_time_or_invalid_timing"],
        "events_missing_source": counts["events_missing_source"],
        "events_missing_valid_peak": counts["events_missing_valid_peak"],
        "cohort_events": len(cohort),
        "source_satellite_counts": dict(Counter(str(event["satellite"]) for event in cohort)),
        "event_status_counts": dict(status_counts),
        "uncapped_background_valid": sum(bool(row["background_uncapped_W_m2"]) for row in event_rows),
        "capped_background_valid": sum(bool(row["background_capped_W_m2"]) for row in event_rows),
        "start_sample_valid": sum(row["start_sample_status"] == "valid" for row in event_rows),
        "total_uncapped_retained_minutes": sum(int(row["uncapped_retained_minutes"]) for row in event_rows),
        "total_capped_retained_minutes": sum(int(row["capped_retained_minutes"] or 0) for row in event_rows),
        "events_uncapped_rxfi_negative": sum(float(row["rxfi_uncapped_fractional_excess"]) < 0 for row in event_rows if row["rxfi_uncapped_fractional_excess"]),
        "failed_l2_file_count": 0,
        "note": "RXFI columns use project fractional-excess convention (F_peak - B)/B; NOAA background_irrad is included only for comparison.",
    }
    fingerprint_inputs = {
        "config": config, "input_lock_sha256": actual_lock_sha,
        "catalog_sha256": catalog_record["sha256"],
        "xrs_l2_sha256": sorted(item["sha256"] for item in manifest_files),
    }
    fingerprint = hashlib.sha256(_canonical_bytes(fingerprint_inputs)).hexdigest()
    output_root = Path(output_root or (base / config["output"]["directory"])).resolve()
    run_dir = output_root / fingerprint
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"run output already exists: {run_dir}; pass --overwrite to replace it")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=output_root))
    try:
        with (staging / "event_backgrounds.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(event_rows)
        _write_json(staging / "summary.json", summary)
        _write_json(staging / "input_manifest.json", manifest)
        _write_json(staging / "run_config.json", {
            "fingerprint": fingerprint, "fingerprint_inputs": fingerprint_inputs,
            "config": config, "input_lock": lock,
        })
        if run_dir.exists():
            shutil.rmtree(run_dir)
        staging.replace(run_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_dir
