"""Compare random single-sample reads from the original and rechunked stores."""

# Zarr 2.18.7 ships without stubs, so its dynamic API produces unknown-type noise.
# Keep checking this script's own interfaces while suppressing those dependency-only reports.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import numpy as np
import zarr
from loguru import logger
from zarr.core import Array


def parse_args() -> argparse.Namespace:
    """Parse paths and benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=461)
    return parser.parse_args()


def select_indices(images: Array, samples: int, seed: int) -> list[int]:
    """Select random timestamps from distinct original time chunks."""
    chunk_size = int(images.chunks[0])
    chunk_count = images.shape[0] // chunk_size
    if samples > chunk_count:
        raise ValueError("Need at least one source time chunk per sampled image")
    generator = np.random.default_rng(seed)
    chunks = generator.choice(chunk_count, size=samples, replace=False)
    indices = [
        int(chunk) * chunk_size
        + int(
            generator.integers(
                0, min(chunk_size, images.shape[0] - int(chunk) * chunk_size)
            )
        )
        for chunk in chunks
    ]
    random.Random(seed).shuffle(indices)
    return indices


def measure(images: Array, indices: list[int]) -> float:
    """Read selected all-channel samples and return elapsed seconds."""
    started = time.perf_counter()
    bytes_read = 0
    for index in indices:
        image = np.asarray(images[index], dtype=np.float32)
        if image.shape != images.shape[1:]:
            raise ValueError(f"Unexpected sample shape at index {index}: {image.shape}")
        bytes_read += image.nbytes
    elapsed = time.perf_counter() - started
    logger.info(
        "Read {} images ({} MiB) in {:.3f}s", len(indices), bytes_read / 2**20, elapsed
    )
    return elapsed


def open_images(store_path: Path, year: int) -> Array:
    """Open and type-check a yearly image array."""
    group = zarr.open_group(str(store_path / str(year) / "dataset"), mode="r")
    images = group["images"]
    if not isinstance(images, Array):
        raise TypeError(f"Expected an image array in {store_path}/{year}/dataset")
    return images


def main() -> None:
    """Measure alternating read batches and report medians."""
    args = parse_args()
    source_images = open_images(args.source, args.year)
    optimized_images = open_images(args.optimized, args.year)
    if source_images.shape != optimized_images.shape:
        raise ValueError("Source and optimized image shapes do not match")
    indices = select_indices(source_images, args.samples, args.seed)
    logger.info("year={} indices={}", args.year, indices)
    logger.info(
        "source chunks={} optimized chunks={}",
        source_images.chunks,
        optimized_images.chunks,
    )

    results: dict[str, list[float]] = {"source": [], "optimized": []}
    for repeat in range(args.repeats):
        arrays = (
            [("source", source_images), ("optimized", optimized_images)]
            if repeat % 2 == 0
            else [("optimized", optimized_images), ("source", source_images)]
        )
        for name, images in arrays:
            elapsed = measure(images, indices)
            results[name].append(elapsed)
            logger.info(
                "repeat={} store={} images_per_second={:.2f}",
                repeat + 1,
                name,
                args.samples / elapsed,
            )

    source_median = statistics.median(results["source"])
    optimized_median = statistics.median(results["optimized"])
    logger.info(
        "median_10_image_seconds source={:.3f} optimized={:.3f}; speedup={:.2f}x",
        source_median,
        optimized_median,
        source_median / optimized_median,
    )


if __name__ == "__main__":
    main()
