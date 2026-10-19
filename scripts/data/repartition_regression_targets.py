#!/usr/bin/env python3
"""Repartition existing XRS regression rows into seasonal validation cohorts."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
from loguru import logger

SPLIT_NAMES = ("train", "validation", "leaky_validation", "test")
REQUIRED_TARGET = "max_xflux"
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class DataRow:
    """A source CSV row with its parsed time and original split."""

    values: dict[str, str]
    timestamp: datetime
    source_split: str


@dataclass(frozen=True)
class Window:
    """An inclusive month-day date range used by the split policy."""

    start: str
    end: str

    def contains(self, timestamp: datetime) -> bool:
        """Return whether a timestamp's month and day falls in this window."""
        start = tuple(int(part) for part in self.start.split("-"))
        end = tuple(int(part) for part in self.end.split("-"))
        current = (timestamp.month, timestamp.day)
        return start <= current <= end


def parse_args() -> argparse.Namespace:
    """Parse the split configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/noaa_g17_v222_splitv2.json"),
        help="JSON split configuration.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Audit existing train/test window overlap and update only the manifest.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    return (parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed).astimezone(UTC)


def timestamp_ns(timestamp: datetime) -> int:
    """Convert a UTC timestamp to Unix nanoseconds."""
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = timestamp - epoch
    return (
        (elapsed.days * 86_400 + elapsed.seconds) * NANOSECONDS_PER_SECOND
        + elapsed.microseconds * 1_000
    )


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_rows(source_dir: Path) -> tuple[list[str], list[DataRow], dict[str, str]]:
    """Read the existing train, validation, and calibration CSV rows."""
    rows: list[DataRow] = []
    source_hashes: dict[str, str] = {}
    expected_fields: Sequence[str] | None = None
    for split in SPLIT_NAMES[:-1]:
        path = source_dir / f"{split}.csv"
        source_hashes[path.name] = sha256_file(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if expected_fields is None:
                expected_fields = fields
            elif fields != expected_fields:
                raise ValueError(f"CSV header mismatch in {path}")
            if "timestamp" not in fields or REQUIRED_TARGET not in fields:
                raise ValueError(f"{path} must contain timestamp and {REQUIRED_TARGET}")
            for values in reader:
                rows.append(DataRow(values, parse_timestamp(values["timestamp"]), split))
    if not expected_fields or not rows:
        raise ValueError(f"No source rows found in {source_dir}")
    timestamps = [row.timestamp for row in rows]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Duplicate timestamps found across historical source CSVs")
    return list(expected_fields), rows, source_hashes


def read_zarr_times(zarr_path: Path, years: set[int]) -> set[int]:
    """Load only Zarr time indexes for years represented in the source CSVs."""
    available: set[int] = set()
    for year in sorted(years):
        group_path = zarr_path / str(year) / "dataset"
        if not group_path.exists():
            logger.warning("Zarr group is missing for year {}: {}", year, group_path)
            continue
        group = zarr.open_group(str(group_path), mode="r")
        times = np.asarray(cast(Any, group["time"][:]), dtype=np.int64)
        if len(times) != len({int(value) for value in times}):
            raise ValueError(f"Duplicate Zarr timestamps found in {year}")
        available.update(int(value) for value in times)
    return available


def is_finite_target(row: DataRow) -> bool:
    """Return whether the active target is present and finite."""
    raw = row.values.get(REQUIRED_TARGET, "").strip()
    if not raw:
        return False
    try:
        return math.isfinite(float(raw))
    except ValueError as error:
        raise ValueError(f"Invalid {REQUIRED_TARGET} value at {row.values['timestamp']}") from error


def read_rows(directory: Path, split: str) -> list[DataRow]:
    """Read one generated CSV for the train/test boundary audit."""
    path = directory / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            DataRow(values, parse_timestamp(values["timestamp"]), split)
            for values in reader
        ]


def interval_overlap_audit(
    train_rows: list[DataRow],
    test_rows: list[DataRow],
    available_times: set[int],
    support: timedelta,
) -> dict[str, Any]:
    """Count pairwise intersections between usable train and test supports."""
    usable_train = sorted(
        row.timestamp
        for row in train_rows
        if is_finite_target(row) and timestamp_ns(row.timestamp) in available_times
    )
    usable_test = sorted(
        row.timestamp
        for row in test_rows
        if is_finite_target(row) and timestamp_ns(row.timestamp) in available_times
    )
    overlapping_train = 0
    overlap_pairs = 0
    examples: list[dict[str, str]] = []
    for train_start in usable_train:
        first = bisect.bisect_right(usable_test, train_start - support)
        last = bisect.bisect_left(usable_test, train_start + support)
        count = last - first
        if count:
            overlapping_train += 1
            overlap_pairs += count
            if len(examples) < 10:
                examples.append({
                    "train_timestamp": train_start.isoformat(),
                    "test_timestamp": usable_test[first].isoformat(),
                })
    overlapping_test = sum(
        bisect.bisect_left(usable_train, test_start + support)
        - bisect.bisect_right(usable_train, test_start - support)
        > 0
        for test_start in usable_test
    )
    return {
        "support_hours": support.total_seconds() / 3600,
        "interval": "[timestamp, timestamp + support)",
        "usable_train_samples": len(usable_train),
        "usable_test_samples": len(usable_test),
        "train_samples_with_overlap": overlapping_train,
        "test_samples_with_overlap": overlapping_test,
        "overlapping_interval_pairs": overlap_pairs,
        "example_overlaps": examples,
        "filtered": False,
    }


def assign_cohorts(
    rows: list[DataRow],
    validation_windows: list[Window],
    calibration_windows: list[Window],
    historical_end_year: int,
) -> dict[str, list[DataRow]]:
    """Route historical rows according to inclusive seasonal date windows."""
    cohorts = {name: [] for name in SPLIT_NAMES[:-1]}
    for row in rows:
        if row.timestamp.year > historical_end_year:
            raise ValueError(f"Historical source includes unexpected year: {row.timestamp.year}")
        if any(window.contains(row.timestamp) for window in validation_windows):
            destination = "validation"
        elif any(window.contains(row.timestamp) for window in calibration_windows):
            destination = "leaky_validation"
        else:
            destination = "train"
        cohorts[destination].append(row)
    return cohorts


def remove_calibration_overlaps(
    cohorts: dict[str, list[DataRow]], available_times: set[int], support: timedelta
) -> list[dict[str, str]]:
    """Drop calibration intervals intersecting usable train or validation windows."""
    core_rows = [
        row
        for name in ("train", "validation")
        for row in cohorts[name]
        if is_finite_target(row) and timestamp_ns(row.timestamp) in available_times
    ]
    core_starts = sorted(row.timestamp for row in core_rows)
    retained: list[DataRow] = []
    removed: list[dict[str, str]] = []
    for row in cohorts["leaky_validation"]:
        start = row.timestamp
        end = start + support
        low = start - support
        index = bisect.bisect_right(core_starts, low)
        conflict = core_starts[index] if index < len(core_starts) and core_starts[index] < end else None
        if conflict is None:
            retained.append(row)
        else:
            removed.append({
                "timestamp": row.values["timestamp"],
                "support_start": start.isoformat(),
                "support_end_exclusive": end.isoformat(),
                "overlap_split": "train_or_validation",
                "overlap_timestamp": conflict.isoformat(),
            })
    cohorts["leaky_validation"] = retained
    return removed


def write_csv(path: Path, fields: list[str], rows: list[DataRow]) -> None:
    """Write rows with the original field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(row.values for row in rows)


def copy_test(source_dir: Path, staging_dir: Path) -> str:
    """Copy the existing test file and verify byte-identical preservation."""
    source = source_dir / "test.csv"
    destination = staging_dir / "test.csv"
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise OSError("Copied test.csv checksum does not match the source")
    return source_hash


def row_count(path: Path) -> int:
    """Count CSV data rows without loading the file into memory."""
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def make_manifest(
    output_dir: Path,
    cohorts: dict[str, list[DataRow]],
    removed: list[dict[str, str]],
    source_hashes: dict[str, str],
    test_hash: str,
    available_times: set[int],
    support_hours: int,
) -> dict[str, Any]:
    """Summarize row counts, Zarr matches, date ranges, and provenance."""
    summary: dict[str, Any] = {}
    for split, rows in cohorts.items():
        usable = [row for row in rows if is_finite_target(row)]
        matched = [row for row in usable if timestamp_ns(row.timestamp) in available_times]
        summary[split] = {
            "rows": len(rows),
            "finite_max_xflux": len(usable),
            "finite_max_xflux_with_zarr_image": len(matched),
            "timestamp_min": min((row.timestamp.isoformat() for row in rows), default=None),
            "timestamp_max": max((row.timestamp.isoformat() for row in rows), default=None),
            "output_sha256": sha256_file(output_dir / f"{split}.csv"),
        }
    summary["test"] = {
        "rows": row_count(output_dir / "test.csv"),
        "output_sha256": sha256_file(output_dir / "test.csv"),
        "source_sha256": test_hash,
        "byte_identical_to_source": sha256_file(output_dir / "test.csv") == test_hash,
    }
    return {
        "split_policy": "seasonal_validation_10_day_calibration_max_xflux_support",
        "target_used_for_core_cohort": REQUIRED_TARGET,
        "target_support_hours": support_hours,
        "core_support_interval": "[timestamp, timestamp + support)",
        "source_sha256": source_hashes,
        "splits": summary,
        "calibration_rows_removed_for_overlap": len(removed),
        "removed_calibration_rows": removed,
        "train_test_target_window_overlap": "not filtered; test split preserved byte-for-byte",
    }


def run(config_path: Path) -> None:
    """Build a new split directory without modifying source files."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_dir = Path(config["source_dir"])
    output_dir = Path(config["output_dir"])
    zarr_path = Path(config["zarr_path"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    fields, rows, source_hashes = read_source_rows(source_dir)
    source_hashes["test.csv"] = sha256_file(source_dir / "test.csv")
    historical_end = int(config["historical_end_year"])
    test_start = int(config["test_start_year"])
    if test_start != historical_end + 1:
        raise ValueError("This workflow expects the test era to immediately follow historical years")
    if any(row.timestamp.year > historical_end for row in rows):
        raise ValueError("Historical source CSVs contain rows from the test era")
    cohorts = assign_cohorts(
        rows,
        [Window(**item) for item in config["core_validation_windows"]],
        [Window(**item) for item in config["calibration_windows"]],
        historical_end,
    )
    available_times = read_zarr_times(zarr_path, {row.timestamp.year for row in rows})
    support_hours = int(config["target_support_hours"])
    removed = remove_calibration_overlaps(
        cohorts, available_times, timedelta(hours=support_hours)
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for split in SPLIT_NAMES[:-1]:
            write_csv(staging_dir / f"{split}.csv", fields, cohorts[split])
        test_hash = copy_test(source_dir, staging_dir)
        manifest = make_manifest(
            staging_dir, cohorts, removed, source_hashes, test_hash, available_times, support_hours
        )
        (staging_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir)
        raise
    for split, details in manifest["splits"].items():
        logger.info("{} rows={} finite={} zarr_matched={}", split, details["rows"], details.get("finite_max_xflux", "n/a"), details.get("finite_max_xflux_with_zarr_image", "n/a"))
    logger.warning(
        "Removed {} calibration rows due to 24-hour target support overlap",
        len(removed),
    )
    logger.warning("Train/test target-window overlap is reported but not filtered")
    logger.info("Wrote split files and manifest to {}", output_dir)


def refresh_manifest(config_path: Path) -> None:
    """Add train/test support-overlap counts to an existing split manifest."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = read_rows(output_dir, "train")
    test_rows = read_rows(output_dir, "test")
    available_times = read_zarr_times(
        Path(config["zarr_path"]),
        {row.timestamp.year for row in train_rows + test_rows},
    )
    support = timedelta(hours=int(config["target_support_hours"]))
    manifest["train_test_target_window_overlap"] = interval_overlap_audit(
        train_rows, test_rows, available_times, support
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    logger.info("Updated train/test overlap audit in {}", manifest_path)


def main() -> None:
    """Run the split generation command."""
    args = parse_args()
    if args.refresh_manifest:
        refresh_manifest(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
