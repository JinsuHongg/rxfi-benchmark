"""Shared signed-log and training-split channel normalization helpers."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


NORMALIZATION_NAME = "signed_log1p_train_channel_zscore"
TRANSFORM_NAME = "signed_log1p"
DEFAULT_STATS_PATH = Path("outputs/input_normalization_stats/train_channel_stats.json")
STD_EPSILON = 1e-6


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Apply sign(x) * ln(1 + abs(x)) element-wise."""
    return np.sign(values) * np.log1p(np.abs(values))


def channel_names_from_array(images: Any) -> list[str]:
    """Read channel order from either supported Zarr metadata spelling."""
    attrs = images.attrs
    channel_names = attrs.get("channel_names", attrs.get("channels"))
    if channel_names is None:
        raise ValueError("Zarr images array is missing channel_names/channels metadata")
    return list(channel_names)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_channel_stats(
    stats_path: str | Path,
    expected_channel_names: list[str],
    expected_zarr_path: str | Path | None = None,
    expected_training_index: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(stats_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported normalization stats schema in {path}")
    transform = payload.get("transform", {})
    if (
        payload.get("normalization") != NORMALIZATION_NAME
        or transform.get("name") != TRANSFORM_NAME
        or transform.get("log_base") != "e"
    ):
        raise ValueError(f"Stats in {path} were not computed with {TRANSFORM_NAME}")
    channel_names = payload.get("channel_names")
    if channel_names != expected_channel_names:
        raise ValueError(
            f"Channel order in {path} does not match the Zarr data: "
            f"{channel_names!r} != {expected_channel_names!r}"
        )

    source = payload.get("source", {})
    if source.get("split") != "train":
        raise ValueError(f"Stats in {path} were not computed from the training split")
    if expected_zarr_path is not None:
        recorded_zarr_path = Path(source.get("zarr_path", "")).resolve()
        if recorded_zarr_path != Path(expected_zarr_path).resolve():
            raise ValueError(f"Stats in {path} were computed from a different Zarr path")
    if expected_training_index is not None:
        index_path = Path(expected_training_index)
        expected_hash = source.get("index_csv_sha256")
        if not index_path.is_file() or not expected_hash or file_sha256(index_path) != expected_hash:
            raise ValueError(f"Stats in {path} do not match training index {index_path}")

    mean = np.asarray(payload.get("mean"), dtype=np.float64)
    std = np.asarray(payload.get("std"), dtype=np.float64)
    expected_shape = (len(expected_channel_names),)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(f"Stats in {path} must contain one mean and std per channel")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError(f"Stats in {path} contain invalid means or standard deviations")
    return mean.astype(np.float32), std.astype(np.float32), payload


def normalize_image(
    image: np.ndarray,
    channel_indices: list[int],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
) -> np.ndarray:
    """Transform an image, then standardize selected channels with train stats."""
    transformed = signed_log1p(image[np.asarray(channel_indices)])
    means = channel_mean[np.asarray(channel_indices), None, None]
    stds = channel_std[np.asarray(channel_indices), None, None]
    return (transformed - means) / np.maximum(stds, STD_EPSILON)
