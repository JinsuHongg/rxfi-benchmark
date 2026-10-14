#!/usr/bin/env python3
"""Compute training-split signed-log per-channel image statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from input_normalization import DEFAULT_STATS_PATH, NORMALIZATION_NAME, TRANSFORM_NAME, channel_names_from_array, signed_log1p


DEFAULT_ZARR_PATH = Path("/scratch/users/jhong36/surya-bench-224-v2.zarr")
DEFAULT_INDEX_CSV = Path("/scratch/users/jhong36/data/surya_index/train_targets.csv")


def timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training_timestamps(index_csv: Path) -> list[tuple[str, int]]:
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


def compute_stats(zarr_path: Path, index_csv: Path) -> dict[str, Any]:
    training_rows = read_training_timestamps(index_csv)
    by_year: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for timestamp, stamp in training_rows:
        year = datetime.fromtimestamp(stamp / 1_000_000_000, tz=timezone.utc).year
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
        images, times = group["images"], group["time"]
        this_channel_names = channel_names_from_array(images)
        if channel_names is None:
            channel_names = this_channel_names
            if not channel_names:
                raise ValueError(f"No channel_names metadata in {group_path}")
            mean = np.zeros(len(channel_names), dtype=np.float64)
            m2 = np.zeros(len(channel_names), dtype=np.float64)
        elif this_channel_names != channel_names:
            raise ValueError(f"Channel order differs in Zarr year {year}")
        if len(images.shape) != 4 or images.shape[1:] != (len(channel_names), 224, 224):
            raise ValueError(f"Unexpected images shape in {group_path}: {images.shape}")

        time_values = np.asarray(times[:], dtype=np.int64)
        positions = {int(value): index for index, value in enumerate(time_values)}
        if len(positions) != len(time_values):
            raise ValueError(f"Duplicate Zarr timestamps found in year {year}")
        requested_positions: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for timestamp, stamp in requested_rows:
            position = positions.get(stamp)
            if position is None:
                missing.append(timestamp)
            else:
                requested_positions[position // images.chunks[0]].append((timestamp, position))

        for chunk_index, chunk_rows in sorted(requested_positions.items()):
            chunk_start = chunk_index * images.chunks[0]
            chunk_end = min(chunk_start + images.chunks[0], images.shape[0])
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

    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} training timestamps are missing from the Zarr store; examples: {examples}")
    if matched_samples != len(training_rows) or total_pixels == 0:
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
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args()

    result = compute_stats(args.zarr_path, args.index_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "training_samples": result["source"]["training_samples"], "channels": result["channel_names"]}))


if __name__ == "__main__":
    main()
