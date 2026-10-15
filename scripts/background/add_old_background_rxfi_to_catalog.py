"""Add the original start-capped background and peak/background RXFI to NOAA events."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from paper_background.calculation import (
    calculate_event_backgrounds,
    index_flare_intervals,
)
from paper_background.noaa import cache_path, download_atomic, l2_url, read_l2_year

WINDOW_SECONDS = 86_400
OUTPUT_FIELDS = (
    "old_background_xrsb_flux",
    "old_rxfi",
    "old_background_status",
    "old_start_threshold_xrsb_flux",
    "old_background_valid_sample_count",
    "old_background_other_flare_masked_sample_count",
    "old_background_retained_sample_count",
)


def parse_timestamp(value: str) -> datetime | None:
    """Parse a catalog timestamp and normalize it to UTC."""
    if not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def positive_finite(value: str) -> float | None:
    """Return a finite positive float or None when the source value is invalid."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def satellite_number(value: str) -> int | None:
    """Parse a GOES satellite label such as ``GOES-16``."""
    prefix, separator, number = value.strip().partition("-")
    if prefix != "GOES" or not separator:
        return None
    try:
        return int(number)
    except ValueError:
        return None


def read_catalog(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read NOAA flare rows and validate the fields used by the calculation."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"NOAA catalog has no header: {path}")
        required = {"start_time", "end_time", "flare_id", "time", "xrsb_irrad", "xrsb_irrad_source"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"NOAA catalog is missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def flare_intervals(rows: list[dict[str, str]]) -> list[tuple[int, int, str]]:
    """Return valid inclusive flare intervals in Unix seconds."""
    intervals = []
    for row in rows:
        start, end = parse_timestamp(row.get("start_time", "")), parse_timestamp(row.get("end_time", ""))
        if start is not None and end is not None and end >= start:
            intervals.append((int(start.timestamp()), int(end.timestamp()), row.get("flare_id", "")))
    return intervals


def load_series(
    satellite: int,
    year: int,
    cache_root: Path,
    version: str,
    series_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    input_records: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one annual NOAA L2 file, downloading it when the local cache misses."""
    key = (satellite, year)
    if key in series_cache:
        return series_cache[key]
    destination = cache_path(cache_root, satellite, year, version)
    if destination.is_file():
        record = {
            "path": str(destination),
            "sha256": sha256_path(destination),
            "size_bytes": destination.stat().st_size,
            "downloaded_this_run": False,
        }
    else:
        legacy_path = cache_root / "xrs_l2" / f"g{satellite}" / f"{year}.nc"
        if legacy_path.is_file():
            destination = legacy_path
            record = {
                "path": str(destination),
                "sha256": sha256_path(destination),
                "size_bytes": destination.stat().st_size,
                "downloaded_this_run": False,
                "legacy_cache_layout": True,
            }
        else:
            record = download_atomic(l2_url(satellite, year, version), destination)
            record["downloaded_this_run"] = True
    seconds, flux, flags, metadata = read_l2_year(destination)
    record["sample_count"] = metadata["sample_count"]
    input_records[f"GOES-{satellite}/{year}"] = record
    series_cache[key] = (seconds, flux, flags)
    return series_cache[key]


def sha256_path(path: Path) -> str:
    """Hash a file incrementally so annual NetCDF inputs need not be loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_columns(
    row: dict[str, str],
    satellite: int,
    start: datetime,
    peak: float,
    cache_root: Path,
    version: str,
    series_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    input_records: dict[str, dict[str, Any]],
    indexed_intervals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, str]:
    """Calculate old capped background and peak/background RXFI for one event."""
    start_seconds = int(start.timestamp())
    window_start = start_seconds - WINDOW_SECONDS
    years = range(datetime.fromtimestamp(window_start, UTC).year, start.year + 1)
    parts = [load_series(satellite, year, cache_root, version, series_cache, input_records) for year in years]
    seconds = np.concatenate([part[0] for part in parts])
    flux = np.concatenate([part[1] for part in parts])
    flags = np.concatenate([part[2] for part in parts])
    threshold = None
    sample_status = "missing_exact_sample"
    for year in years:
        samples, values, quality = series_cache[(satellite, year)]
        position = int(np.searchsorted(samples, start_seconds, side="left"))
        if position < len(samples) and int(samples[position]) == start_seconds:
            if np.isfinite(values[position]) and values[position] > 0 and quality[position] == 0:
                threshold, sample_status = float(values[position]), "valid"
            else:
                sample_status = "invalid_quality_or_flux"
            break
    calculation = calculate_event_backgrounds(
        seconds,
        flux,
        flags,
        start_seconds,
        threshold,
        peak,
        (),
        row.get("flare_id", ""),
        indexed_flare_intervals=indexed_intervals,
        start_sample_status=sample_status,
    )
    background = calculation["background_capped"]
    rxfi = peak / background if background is not None and background > 0 else None
    return {
        "old_background_xrsb_flux": format(background, ".12g") if background is not None else "",
        "old_rxfi": format(rxfi, ".12g") if rxfi is not None and math.isfinite(rxfi) else "",
        "old_background_status": calculation["capped_status"],
        "old_start_threshold_xrsb_flux": format(threshold, ".12g") if threshold is not None else "",
        "old_background_valid_sample_count": str(calculation["valid_quality_count"]),
        "old_background_other_flare_masked_sample_count": str(calculation["other_flare_masked_count"]),
        "old_background_retained_sample_count": str(calculation["capped_retained_count"]),
    }


def write_outputs(
    output_path: Path,
    rows: list[dict[str, str]],
    fields: list[str],
    manifest: dict[str, Any],
    overwrite: bool,
) -> None:
    """Atomically write the enriched catalog and its provenance manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def process_catalog_events(
    rows: list[dict[str, str]],
    start_year: int,
    end_year: int,
    cache_root: Path,
    version: str,
    indexed_intervals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """Compute per-event derived columns and return input provenance and status counts."""
    series_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    input_records: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        row.update({field: "" for field in OUTPUT_FIELDS})
        start = parse_timestamp(row.get("start_time", ""))
        if start is None:
            row["old_background_status"] = "missing_or_invalid_start_time"
            counts["missing_or_invalid_start_time"] += 1
            continue
        if not start_year <= start.year <= end_year:
            row["old_background_status"] = "outside_requested_year_range"
            counts["outside_requested_year_range"] += 1
            continue
        satellite = satellite_number(row.get("xrsb_irrad_source", ""))
        peak = positive_finite(row.get("xrsb_irrad", ""))
        if satellite is None:
            row["old_background_status"] = "missing_or_invalid_satellite"
            counts["missing_or_invalid_satellite"] += 1
            continue
        if peak is None:
            row["old_background_status"] = "missing_or_invalid_peak_flux"
            counts["missing_or_invalid_peak_flux"] += 1
            continue
        row.update(event_columns(
            row, satellite, start, peak, cache_root, version,
            series_cache, input_records, indexed_intervals,
        ))
        counts[row["old_background_status"]] += 1
    return input_records, counts


def main() -> None:
    """Enrich a copied NOAA catalog with original event background and RXFI columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/noaa_flare_catalog_old_background_rxfi.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/noaa_paper_background"))
    parser.add_argument("--xrs-version", default="v2-2-1")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year")
    rows, original_fields = read_catalog(args.catalog)
    indexed_intervals = index_flare_intervals(flare_intervals(rows))
    input_records, counts = process_catalog_events(
        rows, args.start_year, args.end_year, args.cache_dir, args.xrs_version, indexed_intervals
    )
    fields = original_fields + [field for field in OUTPUT_FIELDS if field not in original_fields]
    manifest = {
        "method": {
            "window_utc": "[flare_start_time - 24 hours, flare_start_time)",
            "flare_interval_mask": "exclude samples in any other catalog flare start_time through end_time, inclusive",
            "valid_xrs_sample": "finite positive xrsb_flux with xrsb_flag == 0",
            "upper_cap": "xrsb_flux at exact catalog start_time on xrsb_irrad_source satellite, inclusive",
            "background": "arithmetic mean of remaining samples at or below the start-time cap",
            "rxfi": "catalog xrsb_irrad / old_background_xrsb_flux",
            "catalog_rows_processed_by_start_year": [args.start_year, args.end_year],
            "outside_range": "all original rows are retained; derived columns are blank except status",
        },
        "catalog": {"path": str(args.catalog), "sha256": sha256_path(args.catalog)},
        "xrs_l2": {"product": "NOAA science-quality GOES XRS Level 2 avg1m", "version": args.xrs_version, "files": input_records},
        "output": str(args.output),
        "row_count": len(rows),
        "status_counts": dict(counts),
    }
    write_outputs(args.output, rows, fields, manifest, args.overwrite)
    logger.info("Wrote {} rows to {}", len(rows), args.output)
    logger.info("Event status counts: {}", dict(counts))


if __name__ == "__main__":
    main()
