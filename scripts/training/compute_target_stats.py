#!/usr/bin/env python3
"""Compute train-only asinh and z-score statistics for XRS targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
from loguru import logger

TARGET_COLUMNS = (
    "max_xflux",
    "fluence",
    "max_background_deviation_2h_q10",
    "total_background_deviation_2h_q10",
    "max_background_deviation_2h_q50",
    "total_background_deviation_2h_q50",
    "max_background_deviation_24h_q10",
    "total_background_deviation_24h_q10",
    "max_background_deviation_24h_q50",
    "total_background_deviation_24h_q50",
)
DEFAULT_ZARR_PATH = Path("/scratch/users/jhong36/surya-bench-224-interarea.zarr")
DEFAULT_TARGET_CSV = Path("data/derived/train.csv")
DEFAULT_OUTPUT = Path("outputs/target_normalization_stats/train_target_stats.json")


@dataclass(frozen=True)
class TrainingRow:
    """One parsed training row and its target values."""

    timestamp: str
    year: int
    timestamp_ns: int
    targets: dict[str, float | None]


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC.

    Args:
        value: Timestamp stored in the target CSV.

    Returns:
        Timezone-aware timestamp in UTC.
    """
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_target(value: str | None, name: str, row_number: int) -> float | None:
    """Parse one target while preserving missing or non-finite labels.

    Args:
        value: CSV cell value.
        name: Target column name for error context.
        row_number: One-based CSV line number.

    Returns:
        A finite target value, or ``None`` for a missing or non-finite value.

    Raises:
        ValueError: If a non-empty cell is not numeric.
    """
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid value in {name!r} at CSV line {row_number}") from error
    return result if np.isfinite(result) else None


def read_training_rows(path: Path) -> list[TrainingRow]:
    """Read target values and validate the common-cohort input columns.

    Args:
        path: Generated XRS-only training CSV.

    Returns:
        Parsed training rows in source order.

    Raises:
        ValueError: If required columns are absent or timestamps repeat.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"timestamp", *TARGET_COLUMNS}
        if missing := required - fields:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows: list[TrainingRow] = []
        seen: set[tuple[int, int]] = set()
        for row_number, row in enumerate(reader, start=2):
            timestamp = row["timestamp"]
            parsed_timestamp = parse_timestamp(timestamp)
            stamp = int(parsed_timestamp.timestamp() * 1_000_000_000)
            year = parsed_timestamp.year
            identity = (year, stamp)
            if identity in seen:
                raise ValueError(f"Duplicate training timestamp at CSV line {row_number}: {timestamp}")
            seen.add(identity)
            values = {
                name: parse_target(row.get(name), name, row_number)
                for name in TARGET_COLUMNS
            }
            rows.append(TrainingRow(timestamp, year, stamp, values))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def read_zarr_times(
    zarr_path: Path, years: set[int]
) -> tuple[dict[int, set[int]], dict[str, str]]:
    """Read Zarr time indexes for the training years.

    Args:
        zarr_path: Root of the yearly Zarr store.
        years: Years represented in the training CSV.

    Returns:
        Timestamp sets and per-year hashes for the time arrays.
    """
    timestamps_by_year: dict[int, set[int]] = {}
    hashes_by_year: dict[str, str] = {}
    for year in sorted(years):
        group_path = zarr_path / str(year) / "dataset"
        if not group_path.exists():
            timestamps_by_year[year] = set()
            hashes_by_year[str(year)] = "missing"
            continue
        group = zarr.open_group(str(group_path), mode="r")
        times = np.asarray(cast(Any, group["time"][:]), dtype=np.int64)
        values = {int(value) for value in times}
        if len(values) != len(times):
            raise ValueError(f"Duplicate Zarr timestamps found in {year}")
        timestamps_by_year[year] = values
        hashes_by_year[str(year)] = hashlib.sha256(
            times.astype("<i8", copy=False).tobytes()
        ).hexdigest()
    return timestamps_by_year, hashes_by_year


def fit_target_statistics(rows: list[TrainingRow]) -> dict[str, Any]:
    """Fit scaled-asinh and z-score parameters for each target.

    Args:
        rows: Common-cohort rows with finite values for all targets.

    Returns:
        Per-target training statistics.

    Raises:
        ValueError: If any target has no positive scale or nonzero variance.
    """
    result: dict[str, Any] = {}
    for name in TARGET_COLUMNS:
        target_values = [row.targets[name] for row in rows]
        if any(value is None for value in target_values):
            raise ValueError(f"Common-cohort rows contain missing values for {name!r}")
        raw = np.asarray(
            [value for value in target_values if value is not None], dtype=np.float64
        )
        scale = float(np.median(np.abs(raw)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot fit a positive asinh scale for target {name!r}")
        transformed = np.arcsinh(raw / scale)
        mean = float(np.mean(transformed, dtype=np.float64))
        std = float(np.std(transformed, ddof=0, dtype=np.float64))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Cannot fit a positive target standard deviation for {name!r}")
        result[name] = {
            "n_train": len(rows),
            "scale_s": scale,
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "transformed_mean": mean,
            "transformed_std_population": std,
        }
    return result


def compute_stats(target_csv: Path, zarr_path: Path) -> dict[str, Any]:
    """Compute target statistics on the common Zarr-matched train cohort.

    Args:
        target_csv: Generated training split CSV with all ten targets.
        zarr_path: New interarea Zarr root.

    Returns:
        A JSON-serializable statistics artifact.
    """
    rows = read_training_rows(target_csv)
    zarr_times, time_hashes = read_zarr_times(zarr_path, {row.year for row in rows})
    complete_rows = [row for row in rows if all(value is not None for value in row.targets.values())]
    matched_rows = [
        row for row in complete_rows if row.timestamp_ns in zarr_times.get(row.year, set())
    ]
    if not matched_rows:
        raise ValueError("No rows have all ten finite targets and a matching Zarr timestamp")
    result = {
        "schema_version": 1,
        "fit_split": "train",
        "fit_cohort": "all ten targets finite and timestamp present in Zarr time array",
        "source_csv": str(target_csv),
        "source_csv_sha256": sha256_file(target_csv),
        "zarr_path": str(zarr_path),
        "zarr_time_sha256_by_year": time_hashes,
        "cohort": {
            "train_rows": len(rows),
            "all_targets_finite_rows": len(complete_rows),
            "input_timestamp_matches": len(matched_rows),
            "missing_target_rows": len(rows) - len(complete_rows),
            "missing_input_timestamp_rows": len(complete_rows) - len(matched_rows),
        },
        "transform": {
            "name": "asinh_scaled",
            "formula": "z = asinh(y / s)",
            "scale_rule": "s = median(abs(y)) over finite values in the common matched training cohort, including zeros",
            "inverse_formula": "y = s * sinh(z)",
        },
        "normalization": {
            "name": "per_target_zscore",
            "formula": "(z - transformed_mean) / transformed_std_population",
            "standard_deviation_ddof": 0,
        },
        "targets": fit_target_statistics(matched_rows),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    return result


def main() -> None:
    """Parse arguments, compute statistics, and write the JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-targets", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = compute_stats(args.training_targets, args.zarr_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    logger.info(
        "Wrote target statistics to {} for {}/{} common-cohort rows",
        args.output,
        result["cohort"]["input_timestamp_matches"],
        result["cohort"]["train_rows"],
    )


if __name__ == "__main__":
    main()
