#!/usr/bin/env python3
"""Generate XRS-only 24-hour regression targets for the four SuryaBench splits.

The script resolves and downloads the Hugging Face split CSVs, reads NOAA/NCEI
science-quality GOES XRS Level 2 one-minute XRS-B observations, and writes the
four approved targets while preserving source rows and order. Existing
catalog-based target CSVs are backed up under ``data/derived/legacy_catalog_targets``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from scripts.background.paper_background.noaa import (
        InputDownloadError,
        cache_path,
        download_atomic,
        read_l2_year,
        sha256_file,
    )
except ModuleNotFoundError:
    from paper_background.noaa import (
        InputDownloadError,
        cache_path,
        download_atomic,
        read_l2_year,
        sha256_file,
    )


HF_DATASET = "nasa-ibm-ai4science/surya-bench-flare-forecasting"
HF_SPLITS = ("train", "validation", "leaky_validation", "test")
EXPECTED_ROWS = {"train": 74_760, "validation": 3_672, "leaky_validation": 6_048, "test": 43_848}
TARGET_COLUMNS = (
    "max_xflux",
    "fluence",
    "max_background_deviation",
    "total_background_deviation",
)
PRODUCT_VERSIONS: dict[int, str] = {
    **{satellite: "v2-2-1" for satellite in range(8, 16)},
    16: "v2-2-2",
    17: "v2-2-1",
    18: "v2-2-2",
    19: "v2-2-1",
}
NOAA_TABLE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf"
NOAA_LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/"
NOAA_GOES_R_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
WINDOW_SECONDS = 24 * 60 * 60
CADENCE_SECONDS = 60
EXPECTED_SAMPLES_PER_WINDOW = WINDOW_SECONDS // CADENCE_SECONDS


@dataclass(frozen=True)
class XrsYearData:
    """Store record presence separately from usable quality-flagged samples."""

    record_times: np.ndarray
    measured_times: np.ndarray
    valid_times: np.ndarray
    valid_flux: np.ndarray

# NOAA/NCEI GOES Flare Report ReadMe, Appendix A, Table 5 (10 June 2026).
# This schedule is independent of flare events and is used only to choose which
# satellite's time series supplies each minute. Effective times are UTC.
PRIMARY_SECONDARY_SCHEDULE = (
    ("2009-12-01T00:00:00", 14, 11),
    ("2010-09-01T00:00:00", 14, 15),
    ("2010-10-28T00:00:00", 15, 15),
    ("2011-09-01T00:00:00", 15, 14),
    ("2012-10-23T16:00:00", 14, 15),
    ("2012-11-19T16:31:00", 15, 14),
    ("2015-01-26T16:01:00", 15, 13),
    ("2015-05-21T18:00:00", 14, 13),
    ("2015-06-09T16:25:00", 15, 13),
    ("2016-05-03T13:00:00", 13, 14),
    ("2016-05-12T17:30:00", 14, 13),
    ("2016-05-16T17:00:00", 14, 15),
    ("2016-06-09T17:30:00", 15, 13),
    ("2017-08-16T19:15:00", 13, 14),
    ("2017-08-23T21:20:00", 15, 13),
    ("2017-12-12T16:30:00", 15, 14),
    ("2019-12-09T17:30:00", 16, 15),
    ("2021-08-24T00:00:00", 16, 17),
    ("2023-01-04T00:00:00", 16, 18),
    ("2025-04-07T00:00:00", 18, 19),
)


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def epoch_seconds(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


def sha256(path: Path) -> str:
    return sha256_file(path)


def resolve_hf_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/revision/{revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "rxfi-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)
    resolved = record.get("sha")
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError(f"Hugging Face did not return a full commit SHA for revision {revision!r}")
    return resolved


def timestamps_present(available: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Return a mask for requested timestamps present in a sorted array."""
    if not len(available):
        return np.zeros(len(requested), dtype=bool)
    indices = np.searchsorted(available, requested)
    in_bounds = indices < len(available)
    present = np.zeros(len(requested), dtype=bool)
    present[in_bounds] = available[indices[in_bounds]] == requested[in_bounds]
    return present


def valid_flux_at(data: XrsYearData, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flag-zero presence and flux aligned to requested timestamps."""
    indices = np.searchsorted(data.valid_times, requested)
    present = timestamps_present(data.valid_times, requested)
    flux = np.full(len(requested), np.nan, dtype=float)
    flux[present] = data.valid_flux[indices[present]]
    return present, flux


def empty_xrs_year_data() -> XrsYearData:
    """Create an empty annual series for an unavailable satellite file."""
    empty_times = np.empty(0, dtype=np.int64)
    return XrsYearData(empty_times, empty_times, empty_times, np.empty(0, dtype=float))


def scheduled_year_ranges(
    start: int,
    end: int,
) -> Iterator[tuple[int, int, int, int, int]]:
    """Yield year-bounded UTC ranges with their scheduled satellite pair."""
    boundaries = [
        (epoch_seconds(parse_timestamp(row[0])), int(row[1]), int(row[2]))
        for row in PRIMARY_SECONDARY_SCHEDULE
    ]
    active_index = 0
    for index, (effective, _, _) in enumerate(boundaries):
        if effective <= start:
            active_index = index
        else:
            break
    cursor = start
    while cursor < end:
        while active_index + 1 < len(boundaries) and boundaries[active_index + 1][0] <= cursor:
            active_index += 1
        segment_end = min(end, boundaries[active_index + 1][0] if active_index + 1 < len(boundaries) else end)
        for year in range(datetime.fromtimestamp(cursor, UTC).year, datetime.fromtimestamp(segment_end - 1, UTC).year + 1):
            year_start = epoch_seconds(datetime(year, 1, 1, tzinfo=UTC))
            year_end = epoch_seconds(datetime(year + 1, 1, 1, tzinfo=UTC))
            lo, hi = max(cursor, year_start), min(segment_end, year_end)
            if lo < hi:
                primary, secondary = boundaries[active_index][1:]
                yield lo, hi, year, primary, secondary
        cursor = segment_end


def select_segment_samples(
    expected: np.ndarray,
    primary: XrsYearData,
    secondary: XrsYearData,
    primary_satellite: int,
    secondary_satellite: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Select flag-zero samples and describe minutes with no eligible source."""
    primary_record = timestamps_present(primary.record_times, expected)
    primary_measured = timestamps_present(primary.measured_times, expected)
    primary_valid, primary_flux = valid_flux_at(primary, expected)
    secondary_valid, secondary_flux = valid_flux_at(secondary, expected)
    use_primary = primary_valid
    use_secondary = ~primary_valid & secondary_valid
    missing = ~primary_valid & ~secondary_valid
    gap_details = {
        "timestamp": expected[missing],
        "primary_status": np.select(
            [~primary_record, ~primary_measured],
            ["missing_record", "missing_or_invalid_flux"],
            default="nonzero_flag",
        )[missing],
        "secondary_status": np.select(
            [~timestamps_present(secondary.record_times, expected), ~timestamps_present(secondary.measured_times, expected)],
            ["missing_record", "missing_or_invalid_flux"],
            default="nonzero_flag",
        )[missing],
    }
    times = np.concatenate((expected[use_primary], expected[use_secondary]))
    flux = np.concatenate((primary_flux[use_primary], secondary_flux[use_secondary]))
    satellites = np.concatenate((
        np.full(np.count_nonzero(use_primary), primary_satellite, dtype=np.int16),
        np.full(np.count_nonzero(use_secondary), secondary_satellite or primary_satellite, dtype=np.int16),
    ))
    is_primary = np.concatenate((
        np.ones(np.count_nonzero(use_primary), dtype=bool),
        np.zeros(np.count_nonzero(use_secondary), dtype=bool),
    ))
    return times, flux, satellites, is_primary, gap_details


def fetch_hf_splits(revision: str, cache_dir: Path, offline: bool = False) -> tuple[dict[str, Path], dict[str, Any]]:
    if offline and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--offline requires a full 40-character --revision SHA")
    resolved_revision = revision if re.fullmatch(r"[0-9a-f]{40}", revision) else resolve_hf_revision(revision)
    revision_dir = cache_dir / resolved_revision
    paths: dict[str, Path] = {}
    inputs: dict[str, Any] = {"repo_id": HF_DATASET, "requested_revision": revision, "resolved_revision": resolved_revision, "files": {}}
    for split in HF_SPLITS:
        filename = f"{split}.csv"
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{resolved_revision}/{filename}"
        destination = revision_dir / filename
        if not destination.is_file():
            if offline:
                raise FileNotFoundError(f"cached Hugging Face split is missing: {destination}")
            download_atomic(url, destination)
        paths[split] = destination
        inputs["files"][filename] = {"url": url, "size_bytes": destination.stat().st_size, "sha256": sha256(destination)}
    return paths, inputs


def xrs_url_for(satellite: int, year: int, version: str | None = None) -> str:
    """Build the official NOAA/NCEI annual GOES XRS L2 avg1m science URL.

    Args:
        satellite: GOES satellite number.
        year: UTC year represented by the annual file.
        version: Product version override; defaults to the satellite's pinned version.

    Returns:
        Official direct-download URL for the annual file.
    """
    product_version = version or PRODUCT_VERSIONS[satellite]
    filename = f"sci_xrsf-l2-avg1m_g{satellite}_y{year}_{product_version}.nc"
    if satellite <= 15:
        return f"{NOAA_LEGACY_BASE}goes{satellite}/xrsf-l2-avg1m_science/{filename}"
    return f"{NOAA_GOES_R_BASE}goes{satellite}/l2/data/xrsf-l2-avg1m_science/{filename}"


class AnnualXrsCache:
    """Load cached or downloaded annual files using per-satellite versions."""

    def __init__(self, root: Path, lock_path: Path, offline: bool) -> None:
        self.root = root
        self.offline = offline
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        default_version = lock["xrs_l2"].get("version", "v2-2-1")
        self.locked = {
            (
                int(item["satellite"]),
                int(item["year"]),
                str(item.get("version", default_version)),
            ): item
            for item in lock["xrs_l2"]["files"]
        }
        self.series: dict[tuple[int, int], XrsYearData] = {}
        self.input_manifest: dict[str, dict[str, Any]] = {}
        self.unavailable: list[dict[str, Any]] = []

    def load(self, satellite: int, year: int) -> XrsYearData | None:
        key = (satellite, year)
        if key in self.series:
            return self.series[key]
        version = PRODUCT_VERSIONS[satellite]
        locked = self.locked.get((satellite, year, version))
        url = xrs_url_for(satellite, year, version)
        path = cache_path(self.root, satellite, year, version)
        legacy_path = self.root / "xrs_l2" / f"g{satellite}" / f"{year}.nc"
        candidates = [path]
        if locked is not None:
            candidates.append(legacy_path)
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        download_record: dict[str, Any] | None = None
        if existing is not None and locked is not None:
            actual_size = existing.stat().st_size
            actual_sha = sha256(existing)
            if actual_size != int(locked["size_bytes"]) or actual_sha != locked["sha256"]:
                raise RuntimeError(f"cached NOAA file differs from input lock: {existing}")
        elif existing is None:
            if self.offline:
                self.unavailable.append({"satellite": satellite, "year": year, "url": url, "status": "not_cached_offline"})
                return None
            try:
                download_record = download_atomic(url, path)
                existing = path
            except InputDownloadError as exc:
                if "404" in str(exc) or "410" in str(exc):
                    self.unavailable.append({"satellite": satellite, "year": year, "url": url, "status": str(exc)})
                    return None
                raise
        assert existing is not None
        seconds, flux, flags, metadata = read_l2_year(existing)
        on_grid = (seconds % CADENCE_SECONDS) == 0
        measured = on_grid & np.isfinite(flux) & (flux > 0)
        valid = measured & (flags == 0)
        record_times = np.unique(seconds[on_grid])
        measured_times = np.unique(seconds[measured])
        valid_times, valid_flux = seconds[valid], flux[valid]
        if len(valid_times) > 1:
            unique = np.r_[True, valid_times[1:] != valid_times[:-1]]
            valid_times, valid_flux = valid_times[unique], valid_flux[unique]
        series = XrsYearData(record_times, measured_times, valid_times, valid_flux)
        self.series[key] = series
        item = {
            "satellite": satellite,
            "year": year,
            "product_version": version,
            "path": str(existing),
            "url": url,
            "size_bytes": existing.stat().st_size,
            "sha256": sha256(existing),
            "valid_samples": len(valid_times),
            "sample_count": int(metadata["sample_count"]),
            "locked_input": locked is not None,
        }
        if download_record is not None:
            item["retrieved_at_utc"] = download_record["retrieved_at_utc"]
        self.input_manifest.setdefault(f"g{satellite}/{year}", item)
        return series


def selected_observations(
    start: int,
    end: int,
    cache: AnnualXrsCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Select eligible observations and retain uncovered schedule minutes for QC."""
    parts_t: list[np.ndarray] = []
    parts_f: list[np.ndarray] = []
    parts_s: list[np.ndarray] = []
    parts_primary: list[np.ndarray] = []
    gaps: list[dict[str, Any]] = []
    for lo, hi, year, primary_satellite, secondary_satellite in scheduled_year_ranges(start, end):
        expected = np.arange(lo, hi, CADENCE_SECONDS, dtype=np.int64)
        primary_data = cache.load(primary_satellite, year) or empty_xrs_year_data()
        secondary_data = (
            cache.load(secondary_satellite, year)
            if secondary_satellite != primary_satellite
            else None
        ) or empty_xrs_year_data()
        samples = select_segment_samples(
            expected, primary_data, secondary_data, primary_satellite, secondary_satellite
        )
        sample_times, sample_flux, sample_satellites, sample_is_primary, gap_details = samples
        if len(sample_times):
            parts_t.append(sample_times)
            parts_f.append(sample_flux)
            parts_s.append(sample_satellites)
            parts_primary.append(sample_is_primary)
        gaps.extend({
            "timestamp_utc": datetime.fromtimestamp(int(timestamp), UTC).isoformat(),
            "scheduled_primary_satellite": primary_satellite,
            "scheduled_secondary_satellite": secondary_satellite,
            "primary_status": str(primary_status),
            "secondary_status": str(secondary_status),
        } for timestamp, primary_status, secondary_status in zip(
            gap_details["timestamp"], gap_details["primary_status"], gap_details["secondary_status"]
        ))
    if not parts_t:
        empty_t = np.empty(0, dtype=np.int64)
        return empty_t, np.empty(0, dtype=float), np.empty(0, dtype=np.int16), np.empty(0, dtype=bool), gaps
    times = np.concatenate(parts_t)
    flux = np.concatenate(parts_f)
    satellites = np.concatenate(parts_s)
    is_primary = np.concatenate(parts_primary)
    order = np.argsort(times, kind="stable")
    times, flux, satellites, is_primary = times[order], flux[order], satellites[order], is_primary[order]
    unique = np.r_[True, times[1:] != times[:-1]]
    return times[unique], flux[unique], satellites[unique], is_primary[unique], gaps


def samples_in_window(times: np.ndarray, flux: np.ndarray, satellites: np.ndarray, is_primary: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.searchsorted(times, [start, end])
    return flux[left:right], satellites[left:right], is_primary[left:right]


def lower_half_mean(values: np.ndarray) -> float | None:
    if not len(values):
        return None
    count = (len(values) + 1) // 2
    lower = np.partition(values, count - 1)[:count]
    return float(np.mean(lower))


def coverage(count: int) -> float:
    return float(count / EXPECTED_SAMPLES_PER_WINDOW)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_split(path: Path, split: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise ValueError(f"split file must contain a timestamp column: {path}")
        rows = list(reader)
        columns = list(reader.fieldnames)
    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split} has {len(rows)} rows; expected {EXPECTED_ROWS[split]} from the frozen split")
    seen: set[str] = set()
    previous: datetime | None = None
    for row_number, row in enumerate(rows, start=2):
        timestamp = row["timestamp"]
        if timestamp in seen:
            raise ValueError(f"duplicate timestamp in {path} at CSV line {row_number}: {timestamp}")
        seen.add(timestamp)
        parsed = parse_timestamp(timestamp)
        if previous is not None and parsed <= previous:
            raise ValueError(f"timestamps are not strictly increasing in {path} at CSV line {row_number}")
        previous = parsed
    return columns, rows


def generate_split(
    split: str,
    source_path: Path,
    target_path: Path,
    timestamps: np.ndarray,
    flux: np.ndarray,
    satellites: np.ndarray,
    is_primary: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_columns, source_rows = read_split(source_path, split)
    extra_columns = ["original_row_index", *TARGET_COLUMNS]
    output_columns = source_columns + [column for column in extra_columns if column not in source_columns]
    target_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    missing_prediction = missing_background = 0
    for row_index, source in enumerate(source_rows):
        start = epoch_seconds(parse_timestamp(source["timestamp"]))
        prediction, pred_sat, pred_primary = samples_in_window(timestamps, flux, satellites, is_primary, start, start + WINDOW_SECONDS)
        background, bg_sat, bg_primary = samples_in_window(timestamps, flux, satellites, is_primary, start - WINDOW_SECONDS, start)
        baseline = lower_half_mean(background)
        max_xflux = float(np.max(prediction)) if len(prediction) else None
        fluence = float(CADENCE_SECONDS * np.sum(prediction, dtype=np.float64)) if len(prediction) else None
        if baseline is not None and len(prediction):
            deviations = prediction - baseline
            max_deviation = float(np.max(deviations))
            total_deviation = float(CADENCE_SECONDS * np.sum(deviations, dtype=np.float64))
        else:
            max_deviation = total_deviation = None
        missing_prediction += int(not len(prediction))
        missing_background += int(baseline is None)
        generated = dict(source)
        generated["original_row_index"] = str(row_index)
        generated["max_xflux"] = "" if max_xflux is None else format(max_xflux, ".12g")
        generated["fluence"] = "" if fluence is None else format(fluence, ".12g")
        generated["max_background_deviation"] = "" if max_deviation is None else format(max_deviation, ".12g")
        generated["total_background_deviation"] = "" if total_deviation is None else format(total_deviation, ".12g")
        target_rows.append(generated)
        qc_rows.append({
            "split": split,
            "original_row_index": row_index,
            "timestamp": source["timestamp"],
            "prediction_valid_count": len(prediction),
            "prediction_coverage_fraction": format(coverage(len(prediction)), ".8f"),
            "prediction_primary_count": int(np.count_nonzero(pred_primary)),
            "prediction_secondary_count": int(len(pred_sat) - np.count_nonzero(pred_primary)),
            "background_valid_count": len(background),
            "background_coverage_fraction": format(coverage(len(background)), ".8f"),
            "background_primary_count": int(np.count_nonzero(bg_primary)),
            "background_secondary_count": int(len(bg_sat) - np.count_nonzero(bg_primary)),
            "background_lower50_mean": "" if baseline is None else format(baseline, ".12g"),
            "max_xflux_status": "missing_no_valid_prediction_samples" if max_xflux is None else "ok",
            "background_deviation_status": "missing_no_valid_prediction_samples" if not len(prediction) else ("missing_no_valid_background_samples" if baseline is None else "ok"),
        })
    write_csv(target_path, output_columns, target_rows)
    with target_path.open(encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    if len(result) != len(source_rows):
        raise RuntimeError(f"row count changed for {split}: source={len(source_rows)}, output={len(result)}")
    for index, (src, out) in enumerate(zip(source_rows, result)):
        if out["timestamp"] != src["timestamp"] or int(out["original_row_index"]) != index:
            raise RuntimeError(f"row alignment changed for {split} at row {index}")
        for column in source_columns:
            if out[column] != src[column]:
                raise RuntimeError(f"source column {column} changed for {split} at row {index}")
    summary = {
        "rows": len(source_rows),
        "prediction_windows_without_valid_samples": missing_prediction,
        "background_windows_without_valid_samples": missing_background,
        "valid_max_xflux": sum(row["max_xflux"] != "" for row in target_rows),
        "valid_deviation_targets": sum(row["max_background_deviation"] != "" for row in target_rows),
        "target_sha256": sha256(target_path),
        "source_sha256": sha256(source_path),
    }
    return qc_rows, target_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision or full commit SHA")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/raw/surya_bench_splits_hf"))
    parser.add_argument("--xrs-cache-dir", type=Path, default=Path("data/raw/noaa_paper_background"))
    parser.add_argument("--xrs-input-lock", type=Path, default=Path("configs/paper_background_sensitivity.inputs.lock.json"))
    parser.add_argument("--derived-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xrs_only_target_generation"))
    parser.add_argument("--offline", action="store_true", help="use cached Hugging Face and NOAA inputs only")
    args = parser.parse_args()

    split_paths, hf_inputs = fetch_hf_splits(args.revision, args.hf_cache_dir, offline=args.offline)
    cache = AnnualXrsCache(args.xrs_cache_dir, args.xrs_input_lock, offline=args.offline)

    all_source_rows: dict[str, tuple[list[str], list[dict[str, str]]]] = {
        split: read_split(split_paths[split], split) for split in HF_SPLITS
    }
    all_times = [parse_timestamp(row["timestamp"]) for _, rows in all_source_rows.values() for row in rows]
    span_start = epoch_seconds(min(all_times) - timedelta(hours=48))
    span_end = epoch_seconds(max(all_times) + timedelta(hours=24))
    times, flux, satellites, is_primary, missing_samples = selected_observations(span_start, span_end, cache)

    run_config = {
        "target_columns": list(TARGET_COLUMNS),
        "prediction_window": "[s, s + 24 h)",
        "background_window": "[s - 24 h, s)",
        "background_estimator": "mean of lowest ceil(N/2) valid samples",
        "valid_measurement": "finite positive xrsb_flux and xrsb_flag == 0",
        "missing_samples": "use a flag-zero scheduled secondary sample whenever the primary lacks a flag-zero sample; skip and record minutes when neither satellite has one; stop only if a required prediction or background window has zero valid samples",
        "fluence_and_total_deviation": "60 seconds times valid one-minute sample sum",
        "goes_l2_product_versions_by_satellite": {
            str(satellite): version for satellite, version in PRODUCT_VERSIONS.items()
        },
        "satellite_schedule_source": "NOAA/NCEI GOES Flare Report ReadMe, Appendix A, Table 5, 2026-06-10",
        "satellite_schedule_source_url": NOAA_TABLE_URL,
        "noaa_data_url_templates": {
            "GOES-8_to_15": f"{NOAA_LEGACY_BASE}goes{{satellite}}/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g{{satellite}}_y{{year}}_{{version}}.nc",
            "GOES-16_to_19": f"{NOAA_GOES_R_BASE}goes{{satellite}}/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g{{satellite}}_y{{year}}_{{version}}.nc",
        },
        "satellite_sample_selection": "use flag-zero scheduled primary data, otherwise use flag-zero scheduled secondary data; skip and record minutes when neither satellite has a flag-zero sample",
        "satellite_schedule": [list(row) for row in PRIMARY_SECONDARY_SCHEDULE],
        "noaa_selected_time_start_utc": datetime.fromtimestamp(int(times[0]), UTC).isoformat() if len(times) else None,
        "noaa_selected_time_end_utc": datetime.fromtimestamp(int(times[-1]), UTC).isoformat() if len(times) else None,
    }
    config_json = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
    fingerprint_seed = json.dumps({
        "config": run_config,
        "selection_policy": run_config["satellite_sample_selection"],
        "hf_inputs": hf_inputs,
        "noaa_inputs": sorted(cache.input_manifest.values(), key=lambda item: item["url"]),
        "unavailable_noaa_inputs": sorted(cache.unavailable, key=lambda item: item["url"]),
    }, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(fingerprint_seed.encode()).hexdigest()[:16]
    run_dir = args.output_dir / "runs" / run_id

    # Stage every new output first, then replace the normal derived filenames.
    staged_dir = args.derived_dir / f".xrs_only_staging_{run_id}"
    staged_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hugging_face": hf_inputs,
        "method": run_config,
        "noaa_inputs": list(cache.input_manifest.values()),
        "unavailable_noaa_inputs": cache.unavailable,
        "missing_flag_zero_samples": {
            "count": len(missing_samples),
            "path": str(run_dir / "missing_xrs_samples.csv"),
            "policy": "skip minutes without a flag-zero sample from either scheduled satellite; list each skipped minute for QC",
        },
        "splits": {},
    }
    qc_all: list[dict[str, Any]] = []
    staged_paths: dict[str, Path] = {}
    for split in HF_SPLITS:
        staged_path = staged_dir / f"{split}_targets.csv"
        qc_rows, _, split_summary = generate_split(split, split_paths[split], staged_path, times, flux, satellites, is_primary)
        staged_paths[split] = staged_path
        qc_all.extend(qc_rows)
        summary["splits"][split] = split_summary

    backup_dir = args.derived_dir / "legacy_catalog_targets"
    args.derived_dir.mkdir(parents=True, exist_ok=True)
    for split in HF_SPLITS:
        output = args.derived_dir / f"{split}_targets.csv"
        if output.is_file():
            with output.open(encoding="utf-8", newline="") as handle:
                existing_columns = csv.DictReader(handle).fieldnames or []
            if not all(column in existing_columns for column in TARGET_COLUMNS):
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup = backup_dir / output.name
                if not backup.exists():
                    shutil.copy2(output, backup)
        staged_paths[split].replace(output)
    shutil.rmtree(staged_dir, ignore_errors=True)

    qc_columns = list(qc_all[0]) if qc_all else []
    write_csv(run_dir / "target_coverage_qc.csv", qc_columns, qc_all)
    write_csv(
        run_dir / "missing_xrs_samples.csv",
        ["timestamp_utc", "scheduled_primary_satellite", "scheduled_secondary_satellite", "primary_status", "secondary_status"],
        missing_samples,
    )
    summary["target_coverage_qc"] = str(run_dir / "target_coverage_qc.csv")
    summary["target_coverage_qc_sha256"] = sha256(run_dir / "target_coverage_qc.csv")
    summary["run_config_sha256"] = hashlib.sha256(config_json.encode()).hexdigest()
    (run_dir / "target_generation_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "target_generation_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "derived_dir": str(args.derived_dir), "metadata": str(run_dir / "target_generation_metadata.json"), "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
