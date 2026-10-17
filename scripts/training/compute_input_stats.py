#!/usr/bin/env python3
"""Compute training-split signed-log per-channel image statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from scripts.training.input_normalization import (
    DEFAULT_STATS_PATH,
    NORMALIZATION_NAME,
    TRANSFORM_NAME,
    channel_names_from_array,
    signed_log1p,
)

DEFAULT_ZARR_PATH = Path("/scratch/users/jhong36/surya-bench-224-v2.zarr")
DEFAULT_INDEX_CSV = Path("/scratch/users/jhong36/data/surya_index/train_targets.csv")


def timestamp_ns(value: str) -> int:
    """Convert an ISO timestamp to Unix nanoseconds.

    Args:
        value: Timestamp string from the training index.

    Returns:
        Timestamp expressed as integer nanoseconds since the Unix epoch.
    """
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)


def sha256(path: Path) -> str:
    """Compute a file's SHA-256 digest without loading it all into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training_timestamps(index_csv: Path) -> list[tuple[str, int]]:
    """Read and validate timestamps from a training index CSV.

    Args:
        index_csv: CSV containing the training timestamps.

    Returns:
        Timestamp strings paired with their Unix nanosecond values.

    Raises:
        ValueError: If the timestamp column is absent, rows are empty, or
            duplicate timestamps are present.
    """
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{index_csv} must contain a timestamp column")
        rows = [(row["timestamp"], timestamp_ns(row["timestamp"])) for row in reader]
    if not rows:
        raise ValueError(f"No training rows found in {index_csv}")
    stamps = [stamp for _, stamp in rows]
    if len(stamps) != len(set(stamps)):
        raise ValueError(f"Duplicate timestamps found in training index {index_csv}")
    return rows


def compute_stats(
    zarr_path: Path,
    index_csv: Path,
    allow_missing_timestamps: bool = False,
) -> dict[str, Any]:
    """Compute per-channel population statistics for matched training images.

    Args:
        zarr_path: Root of the year-partitioned Zarr store.
        index_csv: CSV containing training timestamps.
        allow_missing_timestamps: Whether to compute on available image rows
            when some training timestamps are absent from the Zarr store.

    Returns:
        Normalization statistics and source provenance.

    Raises:
        ValueError: If inputs are invalid, channels or image shapes differ,
            timestamps are missing while strict mode is enabled, or no
            training images match the index.
    """
    training_rows = read_training_timestamps(index_csv)
    by_year: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for timestamp, stamp in training_rows:
        year = datetime.fromtimestamp(stamp / 1_000_000_000, tz=UTC).year
        by_year[year].append((timestamp, stamp))

    channel_names: list[str] | None = None
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    total_pixels = 0
    matched_samples = 0
    missing: list[str] = []

    for year, requested_rows in sorted(by_year.items()):
        group_path = zarr_path / str(year) / "dataset"
        if not group_path.exists():
            missing.extend(timestamp for timestamp, _ in requested_rows)
            continue
        group = zarr.open_group(str(group_path), mode="r")
        images = cast(Any, group["images"])
        times = cast(Any, group["time"])
        this_channel_names = channel_names_from_array(images)
        if channel_names is None:
            channel_names = this_channel_names
            if not channel_names:
                raise ValueError(f"No channel_names metadata in {group_path}")
            mean = np.zeros(len(channel_names), dtype=np.float64)
            m2 = np.zeros(len(channel_names), dtype=np.float64)
        elif this_channel_names != channel_names:
            raise ValueError(f"Channel order differs in Zarr year {year}")
        image_shape = tuple(int(size) for size in images.shape)
        if len(image_shape) != 4 or image_shape[1:] != (len(channel_names), 224, 224):
            raise ValueError(f"Unexpected images shape in {group_path}: {image_shape}")

        time_values = np.asarray(times[:], dtype=np.int64)
        positions: dict[int, int] = {
            int(value): int(index) for index, value in enumerate(time_values)
        }
        if len(positions) != len(time_values):
            raise ValueError(f"Duplicate Zarr timestamps found in year {year}")
        chunk_size = int(images.chunks[0])
        requested_positions: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for timestamp, stamp in requested_rows:
            position = positions.get(stamp)
            if position is None:
                missing.append(timestamp)
            else:
                requested_positions[position // chunk_size].append((timestamp, position))

        for chunk_index, chunk_rows in sorted(requested_positions.items()):
            chunk_start = chunk_index * chunk_size
            chunk_end = min(chunk_start + chunk_size, image_shape[0])
            image_chunk = np.asarray(images[chunk_start:chunk_end], dtype=np.float32)
            for timestamp, position in chunk_rows:
                image = image_chunk[position - chunk_start]
                if image.ndim != 3 or image.shape != (len(channel_names), 224, 224):
                    raise ValueError(f"Unexpected image shape for {timestamp}: {image.shape}")
                if not np.isfinite(image).all():
                    raise ValueError(f"Non-finite image values for training timestamp {timestamp}")
                transformed = signed_log1p(image).astype(np.float64, copy=False)
                pixels = int(transformed.shape[1] * transformed.shape[2])
                batch_mean = transformed.mean(axis=(1, 2), dtype=np.float64)
                batch_m2 = np.square(transformed - batch_mean[:, None, None]).sum(axis=(1, 2), dtype=np.float64)
                new_total = total_pixels + pixels
                delta = batch_mean - mean
                m2 += batch_m2 + np.square(delta) * (total_pixels * pixels / new_total)
                mean += delta * (pixels / new_total)
                total_pixels = new_total
                matched_samples += 1

    if missing and not allow_missing_timestamps:
        examples = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} training timestamps are missing from the Zarr store; examples: {examples}")
    if matched_samples == 0 or total_pixels == 0:
        raise ValueError("No complete training-image population was available for statistics")
    assert channel_names is not None and mean is not None and m2 is not None
    std = np.sqrt(np.maximum(m2 / total_pixels, 0.0))

    return {
        "schema_version": 1,
        "normalization": NORMALIZATION_NAME,
        "transform": {
            "name": TRANSFORM_NAME,
            "formula": "sign(x) * ln(1 + abs(x))",
            "log_base": "e",
        },
        "statistic": "population_mean_and_std",
        "std_ddof": 0,
        "channel_names": channel_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count_pixels_per_channel": total_pixels,
        "source": {
            "split": "train",
            "index_csv": str(index_csv),
            "index_csv_sha256": sha256(index_csv),
            "zarr_path": str(zarr_path),
            "training_samples": matched_samples,
            "training_rows": len(training_rows),
            "missing_training_rows": len(missing),
            "missing_training_examples": missing[:5],
        },
        "created_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    """Parse command-line arguments and write the statistics JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument(
        "--allow-missing-timestamps",
        action="store_true",
        help="Compute from matched training images and record unmatched rows.",
    )
    args = parser.parse_args()

    result = compute_stats(
        args.zarr_path,
        args.index_csv,
        allow_missing_timestamps=args.allow_missing_timestamps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "training_samples": result["source"]["training_samples"],
                "training_rows": result["source"]["training_rows"],
                "missing_training_rows": result["source"]["missing_training_rows"],
                "channels": result["channel_names"],
            }
        )
    )


if __name__ == "__main__":
    main()
