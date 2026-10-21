#!/usr/bin/env python3
"""Config-driven ViT-Small quantile regression for NOAA/NCEI targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import matplotlib  # pyright: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import yaml
import zarr
from lightning.pytorch.callbacks import ModelCheckpoint  # pyright: ignore[reportMissingImports]
from lightning.pytorch.loggers import WandbLogger  # pyright: ignore[reportMissingImports]
from loguru import logger
from torch.utils.data import DataLoader, Dataset  # pyright: ignore[reportMissingImports]

from scripts.training.input_normalization import (
    DEFAULT_STATS_PATH,
    NORMALIZATION_NAME,
    channel_names_from_array,
    load_channel_stats,
    normalize_image,
)
from scripts.training.qr_lightning import (
    QuantileRegressionModule,
    TrainingHistoryCallback,
    WarmupAwareEarlyStopping,
)

SPLITS = ("train", "validation", "test")
PRECISIONS = {"32", "16-mixed", "bf16-mixed"}
PREDICTION_FIELDS = [
    "split",
    "original_row_index",
    "timestamp",
    "target_name",
    "target_raw",
    "target_transformed",
    "max_flare_class",
    "q05",
    "q50",
    "q95",
    "q05_raw",
    "q50_raw",
    "q95_raw",
]


@dataclass(frozen=True)
class Record:
    """One valid target row joined to a Zarr image."""

    split: str
    original_row_index: int
    timestamp: str
    year: int
    zarr_index: int
    raw: float
    transformed: float
    flare_class: str
    validation_period: int = -1


def channel_selection(config: dict[str, Any], actual_order: list[str]) -> tuple[list[str], list[int]]:
    """Validate requested channels and return their source positions."""
    selected = config.get("selected_channels", actual_order)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in actual_order for name in selected)
    ):
        raise ValueError("selected_channels must be a unique subset of channel_order")
    indices = [actual_order.index(name) for name in selected]
    if int(config["in_chans"]) != len(indices):
        raise ValueError("in_chans must equal selected channel count")
    return list(selected), indices


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(value: str) -> int:
    """Convert an ISO timestamp to Unix nanoseconds."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)


def has_valid_row_order(rows: list[dict[str, Any]]) -> bool:
    """Accept legacy source order or a chronological split with source indices."""
    indices = [int(row.get("original_row_index", index)) for index, row in enumerate(rows)]
    if indices == list(range(len(rows))):
        return True
    timestamps = [timestamp_ns(str(row["timestamp"])) for row in rows]
    return timestamps == sorted(timestamps)


def validation_period_index(
    timestamp: str, periods: dict[str, dict[str, str]]
) -> int:
    """Map a validation timestamp to the configured seasonal window index."""
    month_day = datetime.fromisoformat(timestamp).strftime("%m-%d")
    for index, period in enumerate(periods.values()):
        if period["start"] <= month_day <= period["end"]:
            return index
    raise ValueError(f"Validation timestamp {timestamp} is outside configured periods")


def validate_transform(spec: Any) -> dict[str, Any]:
    """Validate and normalize a configured target transform."""
    supported = {"log10_scaled", "log10_1p_scaled", "log10_1p", "asinh_zscore"}
    if not isinstance(spec, dict) or spec.get("name") not in supported:
        raise ValueError("Unsupported target_transform.name")
    result: dict[str, Any] = {"name": spec["name"]}
    if spec["name"] == "asinh_zscore":
        for key in ("scale_s", "transformed_mean", "transformed_std_population"):
            value = float(spec.get(key, 0))
            if not math.isfinite(value) or (key != "transformed_mean" and value <= 0):
                raise ValueError(f"Invalid asinh target parameter: {key}")
            result[key] = value
    elif spec["name"] != "log10_1p":
        scale = float(spec.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Scaled transform requires a positive scale")
        result["scale"] = scale
    return result


def forward(raw: float, spec: dict[str, Any]) -> float | None:
    """Transform one raw target, returning None for invalid values."""
    if not math.isfinite(raw):
        return None
    if spec["name"] == "asinh_zscore":
        value = (
            math.asinh(raw / spec["scale_s"]) - spec["transformed_mean"]
        ) / spec["transformed_std_population"]
    elif spec["name"] == "log10_scaled":
        value = math.log10(raw / spec["scale"]) if raw > 0 else math.nan
    elif spec["name"] == "log10_1p_scaled":
        value = math.log10(1 + raw / spec["scale"]) if raw >= 0 else math.nan
    else:
        value = math.log10(1 + raw) if raw >= 0 else math.nan
    return value if math.isfinite(value) else None


def inverse(value: float, spec: dict[str, Any]) -> float:
    """Map a transformed prediction back to target units."""
    if spec["name"] == "asinh_zscore":
        return spec["scale_s"] * math.sinh(
            value * spec["transformed_std_population"] + spec["transformed_mean"]
        )
    if spec["name"] == "log10_scaled":
        return spec["scale"] * 10**value
    if spec["name"] == "log10_1p_scaled":
        return spec["scale"] * (10**value - 1)
    return 10**value - 1


def target(
    row: dict[str, str], config: dict[str, Any], spec: dict[str, Any]
) -> tuple[float, float] | None:
    """Parse and transform a target from one split row."""
    try:
        raw = float(row[config["target_column"]])
    except (KeyError, ValueError):
        return None
    transformed = forward(raw, spec)
    return (raw, transformed) if transformed is not None else None


def read_rows(
    config: dict[str, Any], spec: dict[str, Any]
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Read target splits and verify row alignment and file integrity."""
    result: dict[str, list[dict[str, str]]] = {}
    metadata: dict[str, Any] = {
        "source_split_sha256": {},
        "derived_target_sha256": {},
        "target_validity": {},
    }
    for split in SPLITS:
        if "target_split_dir" in config:
            derived = Path(config["target_split_dir"]) / f"{split}.csv"
            derived_hash = sha256(derived)
            source_hash = "generated_target_split"
            with derived.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "timestamp" not in rows[0] or config["target_column"] not in rows[0]:
                raise RuntimeError(f"{derived} is missing timestamp or target column")
            if not has_valid_row_order(rows):
                raise RuntimeError(f"{split} generated target row order is invalid")
        else:
            source = Path(config["source_split_dir"]) / f"{split}.csv"
            derived = Path(config["derived_target_dir"]) / f"{split}_targets.csv"
            source_hash, derived_hash = sha256(source), sha256(derived)
            with source.open(newline="", encoding="utf-8") as handle:
                source_rows = list(csv.DictReader(handle))
            with derived.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(source_rows) != len(rows):
                raise RuntimeError(f"{split} row count mismatch")
            for index, (source_row, target_row) in enumerate(zip(source_rows, rows)):
                if (
                    source_row["timestamp"] != target_row["timestamp"]
                    or int(target_row.get("original_row_index", index)) != index
                ):
                    raise RuntimeError(f"{split} row alignment mismatch at {index}")
        valid = sum(target(row, config, spec) is not None for row in rows)
        result[split] = rows
        metadata["source_split_sha256"][split] = source_hash
        metadata["derived_target_sha256"][split] = derived_hash
        metadata["target_validity"][split] = {
            "total": len(rows),
            "valid": valid,
            "excluded_missing_or_invalid": len(rows) - valid,
        }
    return result, metadata


def attach_images(
    rows: dict[str, list[dict[str, str]]],
    config: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    """Join valid target rows to exact Zarr timestamps, recording misses."""
    needed: dict[int, set[int]] = defaultdict(set)
    for values in rows.values():
        for row in values:
            if target(row, config, spec) is not None:
                year = datetime.fromisoformat(row["timestamp"]).year
                needed[year].add(timestamp_ns(row["timestamp"]))

    positions: dict[int, dict[int, int]] = {}
    availability: dict[str, Any] = {"per_year": {}, "missing_or_unreadable": []}
    root = Path(config["zarr_path"])
    for year, timestamps in needed.items():
        group_path = root / str(year) / "dataset"
        if not group_path.exists():
            positions[year] = {}
            availability["missing_or_unreadable"].extend(
                {"year": year, "timestamp_ns": stamp} for stamp in timestamps
            )
            continue
        group = zarr.open_group(str(group_path), mode="r")
        images, times = group["images"], group["time"]
        expected_shape = (
            len(config["channel_order"]),
            int(config["image_size"]),
            int(config["image_size"]),
        )
        if (
            channel_names_from_array(images) != config["channel_order"]
            or tuple(images.shape[1:]) != expected_shape
        ):
            raise RuntimeError(f"{year} image schema mismatch")
        found = {
            int(stamp): index
            for index, stamp in enumerate(np.asarray(times[:], dtype=np.int64))
        }
        missing = timestamps - found.keys()
        positions[year] = found
        availability["per_year"][str(year)] = {
            "requested": len(timestamps),
            "available": len(timestamps) - len(missing),
        }
        availability["missing_or_unreadable"].extend(
            {"year": year, "timestamp_ns": stamp} for stamp in missing
        )

    unavailable = {
        (item["year"], item["timestamp_ns"])
        for item in availability["missing_or_unreadable"]
    }
    availability["missing_or_unreadable"] = []
    availability["canonical_channel_order"] = config["channel_order"]
    selected, indices = channel_selection(config, config["channel_order"])
    availability["selected_channel_names"] = selected
    availability["selected_channel_indices"] = indices
    records: dict[str, list[Record]] = {}
    for split, values in rows.items():
        records[split] = []
        for index, row in enumerate(values):
            transformed_target = target(row, config, spec)
            if transformed_target is None:
                continue
            raw, transformed = transformed_target
            timestamp = row["timestamp"]
            year = datetime.fromisoformat(timestamp).year
            stamp = timestamp_ns(timestamp)
            if (year, stamp) in unavailable:
                availability["missing_or_unreadable"].append(
                    {
                        "split": split,
                        "timestamp": timestamp,
                        "year": year,
                        "reason": "timestamp_not_in_zarr",
                    }
                )
                continue
            records[split].append(
                Record(
                    split=split,
                    original_row_index=int(row.get("original_row_index", index)),
                    timestamp=timestamp,
                    year=year,
                    zarr_index=positions[year][stamp],
                    raw=raw,
                    transformed=transformed,
                    flare_class=row.get("max_flare_class", row.get("max_goes_class", "")),
                    validation_period=(
                        validation_period_index(
                            timestamp, config["validation_periods"]
                        )
                        if split == "validation" and config.get("validation_periods")
                        else -1
                    ),
                )
            )
    return records, availability


class SuryaDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Read and normalize one image on demand from the yearly Zarr groups."""

    def __init__(
        self,
        records: list[Record],
        zarr_path: str,
        channel_indices: list[int],
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ) -> None:
        """Store compact sample references and train-fitted normalization."""
        self.records = records
        self.zarr_path = Path(zarr_path)
        self.channel_indices = channel_indices
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.arrays: dict[int, Any] = {}

    def __len__(self) -> int:
        """Return the number of valid image-target pairs."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one image, target, and validation-window identifier."""
        record = self.records[index]
        if record.year not in self.arrays:
            path = self.zarr_path / str(record.year) / "dataset"
            self.arrays[record.year] = zarr.open_group(str(path), mode="r")["images"]
        image = np.asarray(self.arrays[record.year][record.zarr_index], dtype=np.float32)
        expected_shape = (13, 224, 224)
        if image.shape != expected_shape or not np.isfinite(image).all():
            raise RuntimeError(f"Unreadable image {record.timestamp}")
        normalized = normalize_image(
            image, self.channel_indices, self.channel_mean, self.channel_std
        )
        return (
            torch.from_numpy(normalized),
            torch.tensor(record.transformed, dtype=torch.float32),
            torch.tensor(record.validation_period, dtype=torch.int64),
        )


class SuryaDataModule(pl.LightningDataModule):
    """Prepare stable train, validation, and test cohorts from project data."""

    def __init__(self, config: dict[str, Any], target_transform: dict[str, Any]) -> None:
        """Store resolved settings for the shared Zarr-backed data loaders."""
        super().__init__()
        self.config = config
        self.target_transform = target_transform
        self.records: dict[str, list[Record]] = {}
        self.datasets: dict[str, SuryaDataset] = {}
        self.integrity: dict[str, Any] = {}
        self.availability: dict[str, Any] = {}
        self.cohort: dict[str, Any] = {}
        self.stats_metadata: dict[str, Any] = {}

    def setup(self, stage: str | None = None) -> None:
        """Load split/Zarr associations once for all Lightning stages."""
        del stage
        if self.datasets:
            return
        rows, self.integrity = read_rows(self.config, self.target_transform)
        self.records, self.availability = attach_images(
            rows, self.config, self.target_transform
        )
        training_index = None
        if "derived_target_dir" in self.config:
            training_index = Path(self.config["derived_target_dir"]) / "train_targets.csv"
        channel_mean, channel_std, self.stats_metadata = load_channel_stats(
            self.config["normalization_stats_path"],
            self.availability["canonical_channel_order"],
            self.config["zarr_path"],
            training_index,
        )
        self.datasets = {
            split: SuryaDataset(
                self.records[split],
                self.config["zarr_path"],
                self.availability["selected_channel_indices"],
                channel_mean,
                channel_std,
            )
            for split in SPLITS
        }
        self.cohort = {
            split: {
                "total_split_rows": self.integrity["target_validity"][split]["total"],
                "valid_target_rows": self.integrity["target_validity"][split]["valid"],
                "image_target_available_rows": len(self.records[split]),
                "missing_target_rows": self.integrity["target_validity"][split][
                    "excluded_missing_or_invalid"
                ],
                "missing_image_rows": self.integrity["target_validity"][split][
                    "valid"
                ]
                - len(self.records[split]),
            }
            for split in SPLITS
        }

    def _loader(
        self, split: str
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create a loader that shuffles only the training cohort."""
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(self.config["num_workers"]),
            pin_memory=True,
            persistent_workers=int(self.config["num_workers"]) > 0,
        )

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the shuffled training loader."""
        return self._loader("train")

    def val_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the ordered validation loader."""
        return self._loader("validation")

    def test_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return the ordered test loader."""
        return self._loader("test")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Check frozen experiment settings and scheduler configuration."""
    spec = validate_transform(config.get("target_transform"))
    required = {
        "experiment_name",
        "output_dir",
        "target_column",
        "quantiles",
        "model_name",
        "checkpoint_metric",
        "precision",
        "batch_size",
        "num_workers",
        "gradient_accumulation_steps",
        "seed",
        "lr_policy",
        "learning_rate",
        "weight_decay",
    }
    if config.get("normalization") != NORMALIZATION_NAME:
        raise ValueError(f"normalization must be {NORMALIZATION_NAME!r}")
    config.setdefault("normalization_stats_path", str(DEFAULT_STATS_PATH))
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    frozen = {
        "image_size": 224,
        "in_chans": int(config["in_chans"]),
        "channels": int(config["in_chans"]),
        "pretrained": False,
        "quantiles": [0.05, 0.5, 0.95],
        "optimizer": "adamw",
        "checkpoint_metric": "validation_pinball_loss",
    }
    if frozen.keys() - config.keys():
        raise ValueError(f"Missing frozen config fields: {sorted(frozen.keys() - config.keys())}")
    if any(config[key] != value for key, value in frozen.items()):
        raise ValueError("Model, optimizer, or checkpoint settings differ from the QR family")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative")
    if config["model_name"] not in {
        "vit_small_patch16_224",
        "vit_tiny_patch16_224",
        "compact_cnn_v1",
        "alexnet",
        "alexnet_small_head",
        "alexnet_global_pool",
    }:
        raise ValueError("model_name must be a supported 224px model")
    if config["model_name"] == "compact_cnn_v1":
        channels = config.get("cnn_channels")
        if not isinstance(channels, list) or not channels or any(
            int(channel) <= 0 for channel in channels
        ):
            raise ValueError("cnn_channels must be a non-empty list of positive integers")
        stem_kernel_size = int(config.get("cnn_stem_kernel_size", 0))
        if stem_kernel_size < 3 or stem_kernel_size % 2 == 0:
            raise ValueError("cnn_stem_kernel_size must be an odd integer of at least 3")
        dropout = float(config.get("cnn_dropout", -1))
        if not 0 <= dropout < 1:
            raise ValueError("cnn_dropout must be in [0, 1)")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay cannot be negative")
    patience = config.get("early_stopping_patience")
    start_epoch = int(config.get("early_stopping_start_epoch", 1))
    if patience is not None and int(patience) <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if start_epoch <= 0:
        raise ValueError("early_stopping_start_epoch must be positive")
    if config["lr_policy"] not in {"constant", "cosine_warmup"}:
        raise ValueError("lr_policy must be constant or cosine_warmup")
    if config["lr_policy"] == "cosine_warmup":
        ratio = float(config.get("warmup_ratio", 0))
        if not 0 < ratio <= 1:
            raise ValueError("warmup_ratio must be in (0, 1]")
    if config["in_chans"] != config["channels"]:
        raise ValueError("channels must equal in_chans")
    if (
        int(config["batch_size"]) <= 0
        or int(config["num_workers"]) < 0
        or int(config["gradient_accumulation_steps"]) <= 0
    ):
        raise ValueError("Batch size and accumulation must be positive; workers cannot be negative")
    if config["precision"] not in PRECISIONS:
        raise ValueError("Unsupported precision")
    return spec


def resolve_target_transform(config: dict[str, Any]) -> dict[str, Any]:
    """Load the train-fitted asinh parameters for the configured target."""
    if config.get("target_transform", {}).get("name") != "asinh_zscore":
        return config["target_transform"]
    stats_path = Path(config["target_stats_path"])
    with stats_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or payload.get("fit_split") != "train":
        raise ValueError(f"Unsupported or non-train target statistics in {stats_path}")
    if Path(payload.get("zarr_path", "")).resolve() != Path(config["zarr_path"]).resolve():
        raise ValueError("Target statistics were computed from a different Zarr path")
    if "target_split_dir" in config:
        train_csv = Path(config["target_split_dir"]) / "train.csv"
        if sha256(train_csv) != payload.get("source_csv_sha256"):
            raise ValueError("Target statistics do not match configured train.csv")
    try:
        target_stats = payload["targets"][config["target_column"]]
    except KeyError as error:
        raise ValueError(f"No target statistics for {config['target_column']!r}") from error
    return {"name": "asinh_zscore", **target_stats}


def write_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None
) -> None:
    """Write UTF-8 CSV output with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, history: list[dict[str, int | float]]) -> None:
    """Save train and validation loss/error curves as a local SVG."""
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    train_rows = [row for row in history if "train_pinball_loss" in row]
    axes[0].plot(
        [row["epoch"] for row in train_rows],
        [row["train_pinball_loss"] for row in train_rows],
        label="train",
    )
    axes[0].plot(
        epochs,
        [row["validation_pinball_loss"] for row in history],
        label="validation",
    )
    axes[0].set(xlabel="epoch", ylabel="pinball loss")
    axes[0].legend()
    axes[1].plot(
        epochs, [row["validation_mae_q50"] for row in history], label="MAE q50"
    )
    axes[1].plot(
        epochs, [row["validation_rmse_q50"] for row in history], label="RMSE q50"
    )
    axes[1].set(xlabel="epoch", ylabel="transformed-target error")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)


def prediction_rows(
    predictions: list[torch.Tensor],
    records: list[Record],
    config: dict[str, Any],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pair ordered Lightning predictions with source records for CSV export."""
    values = torch.cat(predictions).numpy()
    if len(values) != len(records):
        raise RuntimeError("Prediction count does not match source cohort")
    rows: list[dict[str, Any]] = []
    for prediction, record in zip(values, records):
        raw = [inverse(float(value), spec) for value in prediction]
        rows.append(
            {
                "split": record.split,
                "original_row_index": record.original_row_index,
                "timestamp": record.timestamp,
                "target_name": config["target_column"],
                "target_raw": format(record.raw, ".12g"),
                "target_transformed": format(record.transformed, ".12g"),
                "max_flare_class": record.flare_class,
                "q05": format(float(prediction[0]), ".12g"),
                "q50": format(float(prediction[1]), ".12g"),
                "q95": format(float(prediction[2]), ".12g"),
                "q05_raw": format(raw[0], ".12g"),
                "q50_raw": format(raw[1], ".12g"),
                "q95_raw": format(raw[2], ".12g"),
            }
        )
    if not has_valid_row_order(rows):
        raise RuntimeError("Prediction export order differs from the source split")
    return rows


def git_commit() -> str | None:
    """Return the current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    """Parse the CLI options used by Slurm and local runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_small_224_max_peak_flux_qr.yaml")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--skip-sample-predictions", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--validate-transforms", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision", choices=sorted(PRECISIONS))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    return parser.parse_args()


def run_name(config: dict[str, Any]) -> str:
    """Build a timestamped W&B name containing the searched parameters."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
    learning_rate = f"{float(config['learning_rate']):.0e}"
    learning_rate = learning_rate.replace("e-0", "e-").replace("e+0", "e+")
    weight_decay = f"{float(config['weight_decay']):g}"
    return (
        f"{timestamp}-{config['model_name']}-{config['target_column']}"
        f"-lr{learning_rate}-wd{weight_decay}-seed{config['seed']}"
    )


def write_run_metadata(
    output: Path,
    config: dict[str, Any],
    spec: dict[str, Any],
    data: SuryaDataModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Persist resolved config, data integrity, and local cohort summaries."""
    stats_path = Path(config["normalization_stats_path"])
    metadata = {
        "git_commit": git_commit(),
        "config_path": args.config,
        "config": config,
        "target_transform": spec,
        "input_normalization": {
            "name": NORMALIZATION_NAME,
            "stats_path": str(stats_path),
            "stats_source": data.stats_metadata.get("source"),
        },
        "integrity": data.integrity,
        "cohort": data.cohort,
        "legacy_suryabench_label_used_as_target": False,
        "torch_version": torch.__version__,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0),
        "selected_channel_names": data.availability["selected_channel_names"],
        "selected_channel_indices": data.availability["selected_channel_indices"],
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "transform_metadata.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        output / "cohort_summary.csv",
        [{"split": split, **data.cohort[split]} for split in SPLITS],
    )
    write_csv(
        output / "missing_or_unreadable_images.csv",
        data.availability["missing_or_unreadable"],
        ["split", "timestamp", "year", "reason"],
    )
    (output / "prediction_schema.json").write_text(
        json.dumps(
            {"columns": PREDICTION_FIELDS, "quantiles": config["quantiles"], "target_transform": spec},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Fit the experiment and export selected-checkpoint evaluation artifacts."""
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    overrides = {
        "batch_size": args.batch_size,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    config["experiment_name"] = (
        f"xrs_only_{config['model_name']}_{config['target_column']}"
        f"_lr{config['learning_rate']}_wd{config['weight_decay']}_seed{config['seed']}"
    )
    config["target_transform"] = resolve_target_transform(config)
    spec = validate_config(config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be a positive integer")
        config["epochs"] = args.epochs
    if args.validate_transforms:
        logger.info("Target transform validated: {}", spec)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing CPU fallback")

    output = Path(config["output_dir"])
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    data = SuryaDataModule(config, spec)
    data.setup("fit")
    metadata = write_run_metadata(output, config, spec, data, args)
    logger.info("Prepared cohorts: {}", data.cohort)
    if args.smoke_only:
        logger.info("Smoke-only setup completed")
        return

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT must be set")
    log_config = {
        **config,
        "git_commit": metadata["git_commit"],
        "gpu_name": metadata["gpu_name"],
        "torch_version": metadata["torch_version"],
        "cuda_version": metadata["cuda_version"],
        "cohort": data.cohort,
        "input_normalization": metadata["input_normalization"],
        "derived_target_sha256": data.integrity["derived_target_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wandb_group": args.wandb_group,
        "wandb_job_type": args.wandb_job_type,
        "sample_predictions_enabled": not args.skip_sample_predictions,
        "test_evaluation_enabled": not args.skip_test,
    }
    tags = ["xrs-only", config["target_column"]]
    if args.wandb_group:
        tags.append("lr-wd-sweep")
    wandb_logger = WandbLogger(
        entity=entity,
        project=project,
        name=run_name(config),
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        config=log_config,
        save_dir=str(output),
        log_model=False,
        tags=tags,
    )
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best-epoch{epoch:03d}",
        monitor="validation/pinball_loss",
        mode="min",
        save_top_k=1,
    )
    history = TrainingHistoryCallback()
    callbacks: list[pl.Callback] = [checkpoint, history]
    if config.get("early_stopping_patience") is not None:
        callbacks.append(
            WarmupAwareEarlyStopping(
                monitor="validation/pinball_loss",
                patience=int(config["early_stopping_patience"]),
                start_epoch=int(config.get("early_stopping_start_epoch", 1)),
            )
        )
    model = QuantileRegressionModule(config)
    precision = {"32": "32-true", "16-mixed": "16-mixed", "bf16-mixed": "bf16-mixed"}[
        config["precision"]
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=int(config["epochs"]),
        precision=cast(Any, precision),
        accumulate_grad_batches=int(config["gradient_accumulation_steps"]),
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    write_csv(output / "training_log.csv", history.rows)
    plot_curves(output / "learning_curves.svg", history.rows)

    best_checkpoint = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    best_epoch = int(best_checkpoint["epoch"]) + 1
    best_history = next(
        row for row in history.rows if int(row["epoch"]) == best_epoch
    )
    summary_keys = (
        "validation_pinball_loss",
        "validation_rmse_q50",
        "validation_mae_q50",
        "validation_spearman_q50",
        "validation_r2_q50_transformed",
        "validation_interval_coverage_90",
        "validation_jan_feb_pinball_loss",
        "validation_jul_aug_pinball_loss",
    )
    wandb_logger.experiment.summary.update(
        {
            "best_validation_epoch": best_epoch,
            **{
                f"best_{key}": best_history[key]
                for key in summary_keys
                if key in best_history
            },
        }
    )
    if not args.skip_sample_predictions:
        validation_predictions = trainer.predict(
            model, dataloaders=data.val_dataloader(), ckpt_path="best"
        )
        if validation_predictions is None:
            raise RuntimeError("Validation prediction returned no batches")
        write_csv(
            output / "predictions" / "validation_predictions.csv",
            prediction_rows(
                cast(list[torch.Tensor], validation_predictions),
                data.records["validation"],
                config,
                spec,
            ),
            PREDICTION_FIELDS,
        )

    test_results: Sequence[Mapping[str, float]] = []
    if not args.skip_test:
        test_results = trainer.test(
            model, datamodule=data, ckpt_path="best", verbose=True
        )
        if not args.skip_sample_predictions:
            test_predictions = trainer.predict(
                model, dataloaders=data.test_dataloader(), ckpt_path="best"
            )
            if test_predictions is None:
                raise RuntimeError("Test prediction returned no batches")
            write_csv(
                output / "predictions" / "test_predictions.csv",
                prediction_rows(
                    cast(list[torch.Tensor], test_predictions),
                    data.records["test"],
                    config,
                    spec,
                ),
                PREDICTION_FIELDS,
            )
    test_metrics = test_results[0] if test_results else {}
    summary = [
        {
            "split": "validation",
            "selected_epoch": best_epoch,
            "pinball_loss": best_history["validation_pinball_loss"],
            "mae_q50": best_history["validation_mae_q50"],
            "rmse_q50": best_history["validation_rmse_q50"],
        },
    ]
    if not args.skip_test:
        test_summary = {
            "split": "test",
            "selected_epoch": best_epoch,
            **{key.removeprefix("test/"): value for key, value in test_metrics.items()},
        }
        if "mse_q50" in test_summary:
            test_summary["rmse_q50"] = math.sqrt(float(test_summary.pop("mse_q50")))
        summary.append(test_summary)
    write_csv(output / "summary_metrics.csv", summary)
    scheduler_metadata = getattr(model, "scheduler_metadata", {})
    checkpoint_metadata = {
        "best_checkpoint_path": checkpoint.best_model_path,
        "best_validation_epoch": best_epoch,
        "best_validation_pinball_loss": (
            float(checkpoint.best_model_score.cpu())
            if checkpoint.best_model_score is not None
            else None
        ),
        "selection_split": "validation",
        **scheduler_metadata,
    }
    (output / "checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2) + "\n", encoding="utf-8"
    )
    metadata["scheduler"] = scheduler_metadata
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    expected_hashes = {
        split: sha256(Path(config["target_split_dir"]) / f"{split}.csv")
        if "target_split_dir" in config
        else sha256(Path(config["derived_target_dir"]) / f"{split}_targets.csv")
        for split in SPLITS
    }
    if data.integrity["derived_target_sha256"] != expected_hashes:
        raise RuntimeError("Target CSVs changed during training")
    wandb_logger.experiment.finish()
    logger.info(
        "Training completed; best checkpoint: {}",
        checkpoint.best_model_path,
    )


if __name__ == "__main__":
    main()
