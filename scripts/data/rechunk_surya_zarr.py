"""Copy yearly Surya Zarr groups with image chunks optimized for sample reads."""

# Zarr 2.18.7 ships without stubs, so its dynamic API produces unknown-type noise.
# Keep checking this script's own interfaces while suppressing those dependency-only reports.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import zarr
from loguru import logger
from zarr.core import Array
from zarr.hierarchy import Group


def parse_args() -> argparse.Namespace:
    """Parse source, destination, and staging paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    return parser.parse_args()


def copy_attrs(source: Group | Array, destination: Group | Array) -> None:
    """Copy Zarr attributes without changing their values."""
    destination.attrs.update(dict(source.attrs))


def create_array(
    source: Array,
    destination_group: Group,
    name: str,
    chunks: tuple[int, ...],
) -> Array:
    """Create an output array with the source encoding and requested chunks."""
    destination = destination_group.create_dataset(
        name,
        shape=source.shape,
        chunks=chunks,
        dtype=source.dtype,
        compressor=source.compressor,
        filters=source.filters,
        fill_value=source.fill_value,
        order=source.order,
    )
    copy_attrs(source, destination)
    return destination


def copy_array(
    source: Array,
    destination: Array,
    block_size: int,
    array_name: str,
) -> None:
    """Copy and verify an array in bounded first-axis blocks."""
    for start in range(0, source.shape[0], block_size):
        stop = min(start + block_size, source.shape[0])
        selection = (slice(start, stop),) + (slice(None),) * (source.ndim - 1)
        source_block = np.asarray(source[selection])
        destination[selection] = source_block
        copied_block = np.asarray(destination[selection])
        if not np.array_equal(source_block, copied_block, equal_nan=True):
            raise ValueError(f"Copied values differ in {array_name}[{start}:{stop}]")


def image_chunks(source: Array) -> tuple[int, ...]:
    """Choose one sample and every channel per image chunk."""
    if source.ndim != 4 or source.shape[1] != 13:
        raise ValueError(
            f"Expected images shaped [samples, 13, height, width], got {source.shape}"
        )
    return (1, source.shape[1], source.shape[2], source.shape[3])


def copy_year(source_year: Path, staging_year: Path) -> None:
    """Copy one yearly group and consolidate its dataset metadata."""
    source_group = zarr.open_group(str(source_year), mode="r")
    destination_group = zarr.open_group(str(staging_year), mode="w-")
    copy_attrs(source_group, destination_group)

    source_dataset = source_group["dataset"]
    if not isinstance(source_dataset, Group):
        raise TypeError(f"Expected a dataset group at {source_year / 'dataset'}")
    destination_dataset = destination_group.create_group("dataset")
    copy_attrs(source_dataset, destination_dataset)

    for array_name in ("images", "time"):
        source_array = source_dataset[array_name]
        if not isinstance(source_array, Array):
            raise TypeError(f"Expected array {array_name} in {source_year / 'dataset'}")
        chunks = image_chunks(source_array) if array_name == "images" else source_array.chunks
        destination_array = create_array(
            source_array, destination_dataset, array_name, chunks
        )
        copy_array(
            source_array,
            destination_array,
            block_size=source_array.chunks[0],
            array_name=f"{source_year.name}/dataset/{array_name}",
        )
        logger.info(
            "Copied {}: shape={} chunks={} -> {}",
            f"{source_year.name}/dataset/{array_name}",
            source_array.shape,
            source_array.chunks,
            destination_array.chunks,
        )

    dataset_store = zarr.DirectoryStore(str(staging_year / "dataset"))
    _ = zarr.consolidate_metadata(dataset_store)


def validate_paths(source: Path, destination: Path, staging: Path) -> None:
    """Reject unsafe or ambiguous copy locations before creating output."""
    source = source.resolve()
    destination = destination.resolve()
    staging = staging.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source Zarr directory does not exist: {source}")
    if source == destination or source == staging or destination == staging:
        raise ValueError("Source, destination, and staging paths must be distinct")
    if destination.parent != staging.parent:
        raise ValueError("Staging and destination paths must share a parent directory")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"Destination or staging path already exists: {destination}, {staging}"
        )
    staging.parent.mkdir(parents=True, exist_ok=True)


def run(source: Path, destination: Path, staging: Path) -> None:
    """Rechunk all yearly groups, verify copied values, and publish the result."""
    validate_paths(source, destination, staging)
    years = sorted(
        (path for path in source.iterdir() if path.name.isdigit() and (path / ".zgroup").is_file()),
        key=lambda path: int(path.name),
    )
    if not years:
        raise ValueError(f"No yearly Zarr groups found under {source}")

    for source_year in years:
        logger.info("Rechunking year {}", source_year.name)
        copy_year(source_year, staging / source_year.name)

    os.rename(staging, destination)
    logger.info("Validated rechunked store published at {}", destination)


def main() -> None:
    """Run the rechunking operation."""
    args = parse_args()
    run(args.source, args.destination, args.staging)


if __name__ == "__main__":
    main()
