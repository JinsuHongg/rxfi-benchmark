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
        self.series: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.input_manifest: dict[str, dict[str, Any]] = {}
        self.unavailable: list[dict[str, Any]] = []

    def load(self, satellite: int, year: int) -> tuple[np.ndarray, np.ndarray] | None:
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
        valid = np.isfinite(flux) & (flux > 0) & (flags == 0) & ((seconds % CADENCE_SECONDS) == 0)
        seconds = seconds[valid]
        flux = flux[valid]
        if len(seconds) > 1:
            unique = np.r_[True, seconds[1:] != seconds[:-1]]
            seconds, flux = seconds[unique], flux[unique]
        self.series[key] = (seconds, flux)
        item = {
            "satellite": satellite,
            "year": year,
            "product_version": version,
            "path": str(existing),
            "url": url,
            "size_bytes": existing.stat().st_size,
            "sha256": sha256(existing),
            "valid_samples": len(seconds),
            "sample_count": int(metadata["sample_count"]),
            "locked_input": locked is not None,
        }
        if download_record is not None:
            item["retrieved_at_utc"] = download_record["retrieved_at_utc"]
        self.input_manifest.setdefault(f"g{satellite}/{year}", item)
        return seconds, flux


def selected_observations(
    start: int,
    end: int,
    cache: AnnualXrsCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build valid minute samples and reject secondary-only schedule timestamps."""
    boundaries = [(epoch_seconds(parse_timestamp(row[0])), int(row[1]), int(row[2])) for row in PRIMARY_SECONDARY_SCHEDULE]
    active_index = 0
    for index, (effective, _, _) in enumerate(boundaries):
        if effective <= start:
            active_index = index
        else:
            break
    parts_t: list[np.ndarray] = []
    parts_f: list[np.ndarray] = []
    parts_s: list[np.ndarray] = []
    parts_primary: list[np.ndarray] = []
    mismatch_count = 0
    mismatch_examples: list[str] = []
    cursor = start
    while cursor < end:
        while active_index + 1 < len(boundaries) and boundaries[active_index + 1][0] <= cursor:
            active_index += 1
        next_boundary = boundaries[active_index + 1][0] if active_index + 1 < len(boundaries) else end
        segment_end = min(end, next_boundary)
        primary, secondary = boundaries[active_index][1:]
        for year in range(datetime.fromtimestamp(cursor, UTC).year, datetime.fromtimestamp(segment_end - 1, UTC).year + 1):
            year_start = epoch_seconds(datetime(year, 1, 1, tzinfo=UTC))
            year_end = epoch_seconds(datetime(year + 1, 1, 1, tzinfo=UTC))
            lo, hi = max(cursor, year_start), min(segment_end, year_end)
            if hi <= lo:
                continue
            p = cache.load(primary, year)
            if p is None:
                p_times, p_flux = np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
            else:
                left, right = np.searchsorted(p[0], [lo, hi])
                p_times, p_flux = p[0][left:right], p[1][left:right]
            if secondary is not None and secondary != primary:
                s = cache.load(secondary, year)
                if s is not None:
                    left, right = np.searchsorted(s[0], [lo, hi])
                    candidate_t = s[0][left:right]
                    indices = np.searchsorted(p_times, candidate_t)
                    primary_has = (indices < len(p_times))
                    if np.any(primary_has):
                        primary_has[primary_has] &= p_times[indices[primary_has]] == candidate_t[primary_has]
                    secondary_only = candidate_t[~primary_has]
                    mismatch_count += len(secondary_only)
                    for timestamp in secondary_only:
                        if len(mismatch_examples) == 20:
                            break
                        mismatch_examples.append(
                            f"{datetime.fromtimestamp(int(timestamp), UTC).isoformat()}: "
                            f"scheduled primary GOES-{primary} missing; "
                            f"scheduled secondary GOES-{secondary} has valid data"
                        )
            if len(p_times):
                parts_t.append(p_times)
                parts_f.append(p_flux)
                parts_s.append(np.full(len(p_times), primary, dtype=np.int16))
                parts_primary.append(np.ones(len(p_times), dtype=bool))
        cursor = segment_end
    if mismatch_count:
        details = "\n".join(f"  - {example}" for example in mismatch_examples)
        omitted = mismatch_count - len(mismatch_examples)
        if omitted:
            details += f"\n  - ... and {omitted} more mismatch(es)"
        raise RuntimeError(
            "NOAA GOES schedule/data mismatch: scheduled primary samples are "
            f"missing while secondary samples are available at {mismatch_count} "
            f"timestamp(s). Refusing to generate split data.\n{details}"
        )
    if not parts_t:
        empty_t = np.empty(0, dtype=np.int64)
        return empty_t, np.empty(0, dtype=float), np.empty(0, dtype=np.int16), np.empty(0, dtype=bool)
    times = np.concatenate(parts_t)
    flux = np.concatenate(parts_f)
    satellites = np.concatenate(parts_s)
    is_primary = np.concatenate(parts_primary)
    order = np.argsort(times, kind="stable")
    times, flux, satellites, is_primary = times[order], flux[order], satellites[order], is_primary[order]
    unique = np.r_[True, times[1:] != times[:-1]]
    return times[unique], flux[unique], satellites[unique], is_primary[unique]


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
    times, flux, satellites, is_primary = selected_observations(span_start, span_end, cache)
    if not len(times):
        raise RuntimeError("NOAA XRS selection yielded no valid samples for the split date range")

    run_config = {
        "target_columns": list(TARGET_COLUMNS),
        "prediction_window": "[s, s + 24 h)",
        "background_window": "[s - 24 h, s)",
        "background_estimator": "mean of lowest ceil(N/2) valid samples",
        "valid_measurement": "finite positive xrsb_flux and xrsb_flag == 0",
        "missing_samples": "no interpolation; sum valid samples and record coverage",
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
        "satellite_sample_selection": "scheduled primary only; abort if valid secondary data exists without the scheduled primary",
        "satellite_schedule": [list(row) for row in PRIMARY_SECONDARY_SCHEDULE],
        "noaa_selected_time_start_utc": datetime.fromtimestamp(int(times[0]), UTC).isoformat(),
        "noaa_selected_time_end_utc": datetime.fromtimestamp(int(times[-1]), UTC).isoformat(),
    }
    config_json = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
    fingerprint_seed = json.dumps({
        "config": run_config,
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
    summary["target_coverage_qc"] = str(run_dir / "target_coverage_qc.csv")
    summary["target_coverage_qc_sha256"] = sha256(run_dir / "target_coverage_qc.csv")
    summary["run_config_sha256"] = hashlib.sha256(config_json.encode()).hexdigest()
    (run_dir / "target_generation_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "target_generation_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "derived_dir": str(args.derived_dir), "metadata": str(run_dir / "target_generation_metadata.json"), "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
