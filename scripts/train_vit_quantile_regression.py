#!/usr/bin/env python3
"""Train a checkpoint-independent ViT-Small quantile regressor for OCQR.

SuryaBench contributes images, timestamps, and frozen split membership only.
The supervised value is NOAA/NCEI ``z = log10(max_peak_flux)``.  No legacy
SuryaBench class label is loaded as a learning target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import yaml
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validation", "test", "leaky_validation")
SOURCE_HASHES = {
    "train": "2ec7b8f39367f8340a39889bc66525aff303410d7b7ce6c12a55ea346b55e865",
    "validation": "803d2e5584fe9bbe23bc02cbed1b06fb47520e4863c2b22b5f09f9d5c654c658",
    "test": "40ddef01aebe23e5ee460717a08b7392827eacca2852af074d5f1533f59ebd4b",
    "leaky_validation": "03134a82a53891d25761774c5aad52f77e01673195f7cfd28c0dc061bfe5849e",
}


@dataclass(frozen=True)
class Record:
    split: str
    original_row_index: int
    timestamp: str
    year: int
    zarr_index: int
    max_peak_flux: float
    target_z: float
    max_flare_class: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def valid_target(value: str) -> tuple[float, float] | None:
    """Return physical flux and log10 flux; do not impute blanks or zeros."""
    try:
        flux = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(flux) or flux <= 0:
        return None
    target_z = math.log10(flux)
    return (flux, target_z) if math.isfinite(target_z) else None


def read_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    source_dir, target_dir = Path(config["source_split_dir"]), Path(config["derived_target_dir"])
    rows: dict[str, list[dict[str, str]]] = {}
    metadata: dict[str, Any] = {"source_split_sha256": {}, "derived_target_sha256": {}, "row_alignment": {}, "target_validity": {}}
    for split in SPLITS:
        source, derived = source_dir / f"{split}.csv", target_dir / f"{split}_targets.csv"
        source_digest, derived_digest = sha256(source), sha256(derived)
        if source_digest != SOURCE_HASHES[split]:
            raise RuntimeError(f"Frozen {split}.csv SHA-256 changed: {source_digest}")
        with source.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        with derived.open(encoding="utf-8", newline="") as handle:
            derived_rows = list(csv.DictReader(handle))
        if len(source_rows) != len(derived_rows):
            raise RuntimeError(f"{split}: source/derived row count mismatch")
        for index, (source_row, derived_row) in enumerate(zip(source_rows, derived_rows)):
            if source_row["timestamp"] != derived_row["timestamp"]:
                raise RuntimeError(f"{split}: timestamp mismatch at row {index}")
            if int(derived_row.get("original_row_index", index)) != index:
                raise RuntimeError(f"{split}: original_row_index mismatch at row {index}")
        valid = sum(valid_target(row.get(config["target_column"], "")) is not None for row in derived_rows)
        rows[split] = derived_rows
        metadata["source_split_sha256"][split] = source_digest
        metadata["derived_target_sha256"][split] = derived_digest
        metadata["row_alignment"][split] = {"source_rows": len(source_rows), "derived_rows": len(derived_rows), "timestamp_mismatches": 0}
        metadata["target_validity"][split] = {"total": len(derived_rows), "valid": valid, "excluded_missing_or_invalid": len(derived_rows) - valid}
    return rows, metadata


def attach_zarr_indices(rows: dict[str, list[dict[str, str]]], config: dict[str, Any]) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    """Map valid-target rows to a Zarr image without altering source ordering."""
    needed: dict[int, set[int]] = defaultdict(set)
    for split_rows in rows.values():
        for row in split_rows:
            if valid_target(row.get(config["target_column"], "")) is not None:
                needed[datetime.fromisoformat(row["timestamp"]).year].add(timestamp_ns(row["timestamp"]))
    positions: dict[int, dict[int, int]] = {}
    availability: dict[str, Any] = {"per_year": {}, "missing_or_unreadable": [], "channel_order": None}
    root = Path(config["zarr_path"])
    for year, requested in sorted(needed.items()):
        group_path = root / str(year) / "dataset"
        if not group_path.exists():
            positions[year] = {}
            availability["missing_or_unreadable"].extend({"year": year, "timestamp_ns": stamp, "reason": "missing_year_group"} for stamp in requested)
            continue
        group = zarr.open_group(str(group_path), mode="r")
        images, times = group["images"], group["time"]
        channels = list(images.attrs["channel_names"])
        if channels != config["channel_order"]:
            raise RuntimeError(f"{year}: channel order differs from configuration")
        if tuple(images.shape[1:]) != (int(config["in_chans"]), int(config["image_size"]), int(config["image_size"])):
            raise RuntimeError(f"{year}: unexpected image shape {images.shape}")
        found = {int(stamp): index for index, stamp in enumerate(np.asarray(times[:], dtype=np.int64))}
        missing = requested - found.keys()
        positions[year] = found
        availability["per_year"][str(year)] = {"requested": len(requested), "available": len(requested) - len(missing), "duplicate_zarr_times": len(found) != len(times)}
        availability["missing_or_unreadable"].extend({"year": year, "timestamp_ns": stamp, "reason": "timestamp_not_in_zarr"} for stamp in sorted(missing))
        availability["channel_order"] = channels
    unavailable = {(item["year"], item["timestamp_ns"]) for item in availability["missing_or_unreadable"]}
    availability["missing_or_unreadable"] = []
    records: dict[str, list[Record]] = {}
    for split, split_rows in rows.items():
        selected: list[Record] = []
        for index, row in enumerate(split_rows):
            transformed = valid_target(row.get(config["target_column"], ""))
            if transformed is None:
                continue
            flux, target_z = transformed; stamp = timestamp_ns(row["timestamp"]); year = datetime.fromisoformat(row["timestamp"]).year
            if (year, stamp) in unavailable:
                availability["missing_or_unreadable"].append({"split": split, "timestamp": row["timestamp"], "year": year, "reason": "timestamp_not_in_zarr"})
                continue
            selected.append(Record(split, int(row.get("original_row_index", index)), row["timestamp"], year, positions[year][stamp], flux, target_z, row["max_flare_class"]))
        records[split] = selected
    return records, availability


class SuryaQuantileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[Record], zarr_path: str):
        self.records, self.zarr_path, self.arrays = records, Path(zarr_path), {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        if record.year not in self.arrays:
            self.arrays[record.year] = zarr.open_group(str(self.zarr_path / str(record.year) / "dataset"), mode="r")["images"]
        image = np.asarray(self.arrays[record.year][record.zarr_index], dtype=np.float32)
        if image.shape != (13, 224, 224) or not np.isfinite(image).all():
            raise RuntimeError(f"Unreadable/non-finite image: {record.split} {record.timestamp}")
        mean, std = image.mean(axis=(1, 2), keepdims=True), image.std(axis=(1, 2), keepdims=True)
        return torch.from_numpy((image - mean) / np.maximum(std, 1e-6)), torch.tensor(record.target_z, dtype=torch.float32)


def make_loader(dataset: Dataset[Any], config: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle, num_workers=int(config["num_workers"]), pin_memory=True, drop_last=False)


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    errors = targets.unsqueeze(1) - predictions
    losses = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return losses.mean() if reduction == "mean" else losses


def batch_diagnostics(predictions: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor) -> dict[str, float]:
    per_quantile = pinball_loss(predictions, targets, quantiles, reduction="none").mean(dim=0)
    return {"pinball_loss": float(per_quantile.mean()), **{f"pinball_q{int(round(float(q) * 100)):02d}": float(per_quantile[i]) for i, q in enumerate(quantiles)},
            "mae_q50": float(torch.mean(torch.abs(targets - predictions[:, 1]))), "mse_q50": float(torch.mean((targets - predictions[:, 1]) ** 2)),
            "cross_q05_gt_q50": float(torch.mean((predictions[:, 0] > predictions[:, 1]).float())),
            "cross_q50_gt_q95": float(torch.mean((predictions[:, 1] > predictions[:, 2]).float())),
            "cross_q05_gt_q95": float(torch.mean((predictions[:, 0] > predictions[:, 2]).float())),
            **{f"fraction_target_le_q{int(round(float(q) * 100)):02d}": float(torch.mean((targets <= predictions[:, i]).float())) for i, q in enumerate(quantiles)}}


def run_epoch(model: nn.Module, loader: DataLoader[Any], optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler, device: torch.device, quantiles: torch.Tensor, accumulation: int, amp_enabled: bool) -> dict[str, float]:
    training = optimizer is not None; model.train(training); totals: dict[str, float] = defaultdict(float)
    if training: optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            predictions = model(images); loss = pinball_loss(predictions, targets, quantiles)
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite pinball loss at step {step}")
        if training:
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        diagnostics = batch_diagnostics(predictions.detach(), targets.detach(), quantiles)
        for name, value in diagnostics.items(): totals[name] += value * len(targets)
    result = {name: value / len(loader.dataset) for name, value in totals.items()}
    result["rmse_q50"] = math.sqrt(result.pop("mse_q50"))
    return result


def predict(model: nn.Module, loader: DataLoader[Any], records: list[Record], device: torch.device) -> list[dict[str, object]]:
    model.eval(); result: list[dict[str, object]] = []; position = 0
    with torch.no_grad():
        for images, _ in loader:
            predictions = model(images.to(device, non_blocking=True)).detach().cpu().numpy()
            for prediction in predictions:
                record = records[position]; position += 1
                result.append({"split": record.split, "original_row_index": record.original_row_index, "timestamp": record.timestamp,
                               "max_peak_flux": format(record.max_peak_flux, ".12g"), "z": format(record.target_z, ".12g"), "max_flare_class": record.max_flare_class,
                               "q05": format(float(prediction[0]), ".12g"), "q50": format(float(prediction[1]), ".12g"), "q95": format(float(prediction[2]), ".12g")})
    if position != len(records): raise RuntimeError("Prediction row count mismatch")
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def plot_curves(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]; fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_pinball_loss"] for row in history], label="train"); axes[0].plot(epochs, [row["validation_pinball_loss"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="pinball loss", title="Quantile loss"); axes[0].legend()
    axes[1].plot(epochs, [row["validation_mae_q50"] for row in history], label="MAE q50"); axes[1].plot(epochs, [row["validation_rmse_q50"] for row in history], label="RMSE q50")
    axes[1].set(xlabel="epoch", ylabel="z error", title="Median-quantile diagnostics"); axes[1].legend(); fig.tight_layout(); fig.savefig(path, format="svg"); plt.close(fig)


def smoke_test(model: nn.Module, loader: DataLoader[Any], config: dict[str, Any], device: torch.device, quantiles: torch.Tensor, output: Path, amp_enabled: bool) -> dict[str, object]:
    iterations = 32; model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled); losses: list[float] = []; crossing: list[float] = []; gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device); iterator = iter(loader)
    for _ in range(iterations):
        try: images, targets = next(iterator)
        except StopIteration: iterator = iter(loader); images, targets = next(iterator)
        images, targets = images.to(device), targets.to(device); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            predictions = model(images); loss = pinball_loss(predictions, targets, quantiles)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite smoke pinball loss")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradients_finite = gradients_finite and all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None)
        scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach())); crossing.append(float((predictions[:, 0] > predictions[:, 2]).float().mean()))
    torch.cuda.synchronize(device)
    result = {"iterations": iterations, "batch_size": int(config["batch_size"]), "finite_loss": True, "finite_gradients": gradients_finite,
              "first_pinball_loss": losses[0], "last_pinball_loss": losses[-1], "mean_q05_gt_q95_crossing_rate": float(np.mean(crossing)),
              "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2), "precision": config["precision"], "device": torch.cuda.get_device_name(device)}
    (output / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/vit_small_224_max_peak_flux_qr.yaml"); parser.add_argument("--smoke-only", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); required = {"target_column", "target_transform", "quantiles", "model_name", "checkpoint_metric"}
    if not required.issubset(config) or config["target_column"] != "max_peak_flux" or config["target_transform"] != "log10": raise RuntimeError("Unexpected regression target configuration")
    if config["quantiles"] != [0.05, 0.5, 0.95]: raise RuntimeError("Quantiles must be [0.05, 0.50, 0.95]")
    if config["precision"] not in {"fp16", "fp32"}: raise RuntimeError("precision must be fp16 or fp32")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required; refusing CPU fallback")
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True); (output / "checkpoints").mkdir(exist_ok=True); (output / "predictions").mkdir(exist_ok=True)
    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    rows, integrity = read_rows(config); target_hashes_before = dict(integrity["derived_target_sha256"]); records, availability = attach_zarr_indices(rows, config)
    datasets = {split: SuryaQuantileDataset(values, config["zarr_path"]) for split, values in records.items()}; loaders = {split: make_loader(dataset, config, split == "train") for split, dataset in datasets.items()}
    device, quantiles = torch.device("cuda"), torch.tensor(config["quantiles"], dtype=torch.float32, device="cuda"); amp_enabled = config["precision"] == "fp16"
    model = timm.create_model(config["model_name"], pretrained=bool(config["pretrained"]), in_chans=int(config["in_chans"]), num_classes=len(quantiles)).to(device)
    metadata = {"config": config, "integrity": integrity, "image_availability": availability, "record_counts_after_image_availability": {split: len(values) for split, values in records.items()}, "legacy_suryabench_label_used_as_target": False,
                "prediction_column_order": ["q05", "q50", "q95"], "calibration_partition": "unresolved_no_fixed_partition_or_OCQR_implementation_in_repository", "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8"); (output / "dataset_integrity.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "missing_or_unreadable_images.csv", availability["missing_or_unreadable"], ["split", "timestamp", "year", "reason"])
    (output / "prediction_schema.json").write_text(json.dumps({"columns": ["split", "original_row_index", "timestamp", "max_peak_flux", "z", "max_flare_class", "q05", "q50", "q95"], "quantiles": config["quantiles"], "raw_predictions_preserved": True}, indent=2) + "\n", encoding="utf-8")
    smoke = smoke_test(model, loaders["train"], config, device, quantiles, output, amp_enabled); print(json.dumps({"smoke": smoke, "target_validity": integrity["target_validity"]}), flush=True)
    target_hashes_after_smoke = {split: sha256(Path(config["derived_target_dir"]) / f"{split}_targets.csv") for split in SPLITS}
    if target_hashes_before != target_hashes_after_smoke: raise RuntimeError("Derived target files changed during smoke test")
    metadata["derived_target_sha256_after_smoke"] = target_hashes_after_smoke; metadata["derived_targets_unchanged_after_smoke"] = True
    (output / "dataset_integrity.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.smoke_only: return
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model = timm.create_model(config["model_name"], pretrained=False, in_chans=int(config["in_chans"]), num_classes=len(quantiles)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled); history: list[dict[str, float]] = []; best_loss, best_epoch = math.inf, 0
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter(); train = run_epoch(model, loaders["train"], optimizer, scaler, device, quantiles, int(config["gradient_accumulation_steps"]), amp_enabled); validation = run_epoch(model, loaders["validation"], None, scaler, device, quantiles, 1, amp_enabled)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"validation_{key}": value for key, value in validation.items()}, "elapsed_seconds": round(time.perf_counter() - started, 2)}; history.append(row); write_csv(output / "training_log.csv", history); print(json.dumps(row), flush=True)
        if validation["pinball_loss"] < best_loss:
            best_loss, best_epoch = validation["pinball_loss"], epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "quantiles": config["quantiles"], "checkpoint_metric": "validation_pinball_loss", "validation_metrics": validation}, output / "checkpoints" / "best_validation_pinball_loss.pt")
    plot_curves(output / "learning_curves.svg", history); checkpoint = torch.load(output / "checkpoints" / "best_validation_pinball_loss.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    metrics_rows = []
    for split in ("validation", "test", "leaky_validation"):
        metrics = run_epoch(model, loaders[split], None, scaler, device, quantiles, 1, amp_enabled); metrics_rows.append({"split": split, "selected_epoch": best_epoch, **metrics}); write_csv(output / "predictions" / f"{split}_predictions.csv", predict(model, loaders[split], records[split], device))
    write_csv(output / "summary_metrics.csv", metrics_rows); (output / "checkpoint_metadata.json").write_text(json.dumps({"best_checkpoint": "checkpoints/best_validation_pinball_loss.pt", "best_validation_epoch": best_epoch, "best_validation_pinball_loss": best_loss, "selection_split": "validation", "test_not_used_for_selection": True, "smoke": smoke}, indent=2) + "\n", encoding="utf-8")
    target_hashes_after = {split: sha256(Path(config["derived_target_dir"]) / f"{split}_targets.csv") for split in SPLITS}
    if target_hashes_before != target_hashes_after: raise RuntimeError("Derived target files changed during training")
    metadata["derived_target_sha256_after_training"] = target_hashes_after; metadata["derived_targets_unchanged_after_training"] = True
    (output / "dataset_integrity.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
