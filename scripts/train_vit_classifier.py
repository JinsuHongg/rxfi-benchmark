#!/usr/bin/env python3
"""Train the first NOAA-targeted ViT-Small SuryaBench baseline.

SuryaBench columns are used exclusively for timestamp/split membership and image
lookup.  The only supervised target read by this program is the configured
NOAA-derived target column from ``data/derived/*_targets.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zarr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm


SPLITS = ("train", "validation", "test", "leaky_validation")
SOURCE_HASHES = {
    "train": "2ec7b8f39367f8340a39889bc66525aff303410d7b7ce6c12a55ea346b55e865",
    "validation": "803d2e5584fe9bbe23bc02cbed1b06fb47520e4863c2b22b5f09f9d5c654c658",
    "test": "40ddef01aebe23e5ee460717a08b7392827eacca2852af074d5f1533f59ebd4b",
    "leaky_validation": "03134a82a53891d25761774c5aad52f77e01673195f7cfd28c0dc061bfe5849e",
}
CLASS_PATTERN = re.compile(r"^(?:FQ|[ABCMX][0-9]+(?:\.[0-9]+)?)$")
BAND_ORDER = ("FQ", "A", "B", "C", "M", "X")

def flare_class_to_band(value: str) -> str:
    if not isinstance(value, str) or not CLASS_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid NOAA flare class: {value!r}")
    return "FQ" if value == "FQ" else value[0]

def class_mapping_for_representation(representation: str) -> dict[str, int]:
    if representation != "ordinal_band":
        raise ValueError(f"Unsupported fixed representation: {representation}")
    return {label: index for index, label in enumerate(BAND_ORDER)}

def encoded_label(raw: str, config: dict[str, Any]) -> str:
    representation = config.get("target_representation", "exact_class")
    if representation == "exact_class":
        if not CLASS_PATTERN.fullmatch(raw): raise ValueError(f"Invalid NOAA flare class: {raw!r}")
        return raw
    if representation == "ordinal_band": return flare_class_to_band(raw)
    raise ValueError(f"Unsupported target_representation: {representation}")
PRECISIONS = {"32", "16-mixed", "bf16-mixed"}
ACTIVE_PRECISION = "32"

def channel_selection(config, actual_order):
    selected = config.get("selected_channels", actual_order)
    if not selected or len(set(selected)) != len(selected) or any(name not in actual_order for name in selected):
        raise ValueError("selected_channels must be a non-empty, unique subset of canonical channel_order")
    indices = [actual_order.index(name) for name in selected]
    if int(config["in_chans"]) != len(indices):
        raise ValueError("in_chans must equal selected channel count")
    return list(selected), indices


@dataclass(frozen=True)
class Record:
    split: str
    timestamp: str
    year: int
    zarr_index: int
    target: str
    target_index: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def class_sort_key(label: str) -> tuple[int, float, str]:
    if label == "FQ":
        return (0, 0.0, label)
    letters = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    return (letters[label[0]], float(label[1:]), label)


def load_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    target_dir = Path(config["derived_target_dir"])
    source_dir = Path(config["source_split_dir"])
    rows: dict[str, list[dict[str, str]]] = {}
    integrity: dict[str, Any] = {"source_split_sha256": {}, "row_alignment": {}}
    for split in SPLITS:
        source = source_dir / f"{split}.csv"
        derived = target_dir / f"{split}_targets.csv"
        if not source.exists() or not derived.exists():
            raise FileNotFoundError(f"Missing source or derived file for {split}: {source}, {derived}")
        source_hash = sha256(source)
        if source_hash != SOURCE_HASHES[split]:
            raise RuntimeError(f"Frozen {split}.csv SHA-256 changed: {source_hash}")
        with source.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        with derived.open(encoding="utf-8", newline="") as handle:
            derived_rows = list(csv.DictReader(handle))
        if len(source_rows) != len(derived_rows):
            raise RuntimeError(f"{split}: source/derived row counts differ")
        mismatches = [i for i, (a, b) in enumerate(zip(source_rows, derived_rows)) if a["timestamp"] != b["timestamp"]]
        if mismatches:
            raise RuntimeError(f"{split}: timestamp alignment failed at rows {mismatches[:5]}")
        invalid = [r[config["target_column"]] for r in derived_rows if not CLASS_PATTERN.match(r.get(config["target_column"], ""))]
        if invalid:
            raise RuntimeError(f"{split}: invalid NOAA class labels: {invalid[:5]}")
        rows[split] = derived_rows
        integrity["source_split_sha256"][split] = source_hash
        integrity["row_alignment"][split] = {"source_rows": len(source_rows), "derived_rows": len(derived_rows), "timestamp_mismatches": 0}
    return rows, integrity


def attach_zarr_indices(rows: dict[str, list[dict[str, str]]], config: dict[str, Any], mapping: dict[str, int]) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    needed: dict[int, set[int]] = defaultdict(set)
    for split_rows in rows.values():
        for row in split_rows:
            needed[datetime.fromisoformat(row["timestamp"]).year].add(timestamp_ns(row["timestamp"]))
    positions: dict[int, dict[int, int]] = {}
    zarr_path = Path(config["zarr_path"])
    channel_order: list[str] | None = None
    availability: dict[str, Any] = {"per_year": {}, "missing_or_unreadable": []}
    for year, requested in sorted(needed.items()):
        group_path = zarr_path / str(year) / "dataset"
        if not group_path.exists():
            availability["missing_or_unreadable"].extend({"year": year, "timestamp_ns": value, "reason": "missing_year_group"} for value in requested)
            continue
        group = zarr.open_group(str(group_path), mode="r")
        images, times = group["images"], group["time"]
        actual_order = list(images.attrs["channel_names"])
        if channel_order is None:
            channel_order = actual_order
        if actual_order != channel_order or actual_order != config["channel_order"]:
            raise RuntimeError(f"{year}: channel order differs from config: {actual_order}")
        if images.shape[1:] != (len(actual_order), config["image_size"], config["image_size"]):
            raise RuntimeError(f"{year}: unexpected image shape {images.shape}")
        found = {int(value): index for index, value in enumerate(np.asarray(times[:], dtype=np.int64))}
        duplicate_times = len(found) != int(times.shape[0])
        missing = sorted(requested - found.keys())
        availability["missing_or_unreadable"].extend({"year": year, "timestamp_ns": value, "reason": "timestamp_not_in_zarr"} for value in missing)
        positions[year] = found
        availability["per_year"][str(year)] = {"requested": len(requested), "available": len(requested) - len(missing), "duplicate_zarr_times": duplicate_times}
    unavailable = {(item["year"], item["timestamp_ns"]) for item in availability["missing_or_unreadable"]}
    # Replace year-level diagnostics with split-level records suitable for audit CSV export.
    availability["missing_or_unreadable"] = []
    result: dict[str, list[Record]] = {}
    seen: set[tuple[str, int]] = set()
    for split, split_rows in rows.items():
        records: list[Record] = []
        for row in split_rows:
            stamp = timestamp_ns(row["timestamp"])
            year = datetime.fromisoformat(row["timestamp"]).year
            key = (split, stamp)
            if key in seen:
                raise RuntimeError(f"Duplicate sample loading key in {split}: {row['timestamp']}")
            seen.add(key)
            if (year, stamp) in unavailable:
                availability["missing_or_unreadable"].append({"split": split, "timestamp": row["timestamp"], "year": year, "reason": "timestamp_not_in_zarr"})
                continue
            label = encoded_label(row[config["target_column"]], config)
            records.append(Record(split, row["timestamp"], year, positions[year][stamp], label, mapping[label]))
        result[split] = records
    selected_names, selected_indices = channel_selection(config, channel_order or config["channel_order"])
    availability["canonical_channel_order"] = channel_order
    availability["selected_channel_names"] = selected_names
    availability["selected_channel_indices"] = selected_indices
    return result, availability


class SuryaDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[Record], zarr_path: str, channel_indices: list[int]) -> None:
        self.records, self.zarr_path, self.channel_indices = records, Path(zarr_path), channel_indices
        self.arrays: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        if record.year not in self.arrays:
            self.arrays[record.year] = zarr.open_group(str(self.zarr_path / str(record.year) / "dataset"), mode="r")["images"]
        image = np.asarray(self.arrays[record.year][record.zarr_index], dtype=np.float32)
        if image.shape != (13, 224, 224) or not np.isfinite(image).all():
            raise RuntimeError(f"Unreadable/non-finite image: split={record.split} timestamp={record.timestamp}")
        image = image[self.channel_indices]
        # Training-only, per-image/channel normalization avoids leakage and applies to each selected actual channel.
        mean = image.mean(axis=(1, 2), keepdims=True)
        std = image.std(axis=(1, 2), keepdims=True)
        image = (image - mean) / np.maximum(std, 1e-6)
        return torch.from_numpy(image), record.target_index


def make_loader(dataset: Dataset[Any], config: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle, num_workers=int(config["num_workers"]), pin_memory=True, drop_last=False)


def metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    class_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    per_class = [{"class": label, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]), "prediction_count": int(matrix[:, i].sum())} for i, label in enumerate(labels)]
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_f1": float(np.mean(f1)), "n_samples": len(y_true)}
    return summary, per_class, matrix


def run_epoch(model: nn.Module, loader: DataLoader[Any], optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler, device: torch.device, accumulation: int) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    true, pred = [], []
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train), torch.amp.autocast("cuda", dtype=torch.bfloat16 if ACTIVE_PRECISION == "bf16-mixed" else torch.float16, enabled=device.type == "cuda" and ACTIVE_PRECISION != "32"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        if is_train:
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * inputs.shape[0]
        true.extend(targets.detach().cpu().tolist())
        pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), true, pred


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["class"])
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, history: list[dict[str, Any]]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Training and validation loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["validation_macro_f1"] for r in history], marker="o")
    axes[1].set(xlabel="epoch", ylabel="macro F1", title="Validation macro F1")
    fig.tight_layout(); fig.savefig(path, format="svg"); plt.close(fig)


def plot_confusion(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set(title="Test confusion matrix: NOAA-derived max_flare_class", xlabel="predicted class", ylabel="true class")
    ticks = np.arange(len(labels)); ax.set_xticks(ticks, labels, rotation=90, fontsize=5); ax.set_yticks(ticks, labels, fontsize=5)
    fig.tight_layout(); fig.savefig(path, format="svg"); plt.close(fig)


def smoke_test(model: nn.Module, loader: DataLoader[Any], config: dict[str, Any], device: torch.device, output: Path) -> dict[str, Any]:
    model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and ACTIVE_PRECISION == "16-mixed")
    criterion = nn.CrossEntropyLoss(); losses = []
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    for step in range(20):
        try: inputs, targets = next(iterator)
        except StopIteration: iterator = iter(loader); inputs, targets = next(iterator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if ACTIVE_PRECISION == "bf16-mixed" else torch.float16, enabled=device.type == "cuda" and ACTIVE_PRECISION != "32"):
            loss = criterion(model(inputs), targets)
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite smoke loss at step {step}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    result = {"iterations": 20, "batch_size": int(config["batch_size"]), "finite_loss": True, "first_loss": losses[0], "last_loss": losses[-1], "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2), "device": torch.cuda.get_device_name(device)}
    (output / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    global ACTIVE_PRECISION
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_small_224_max_flare_class.yaml")
    parser.add_argument("--smoke-only", action="store_true"); parser.add_argument("--batch-size", type=int); parser.add_argument("--precision", choices=sorted(PRECISIONS)); parser.add_argument("--num-workers", type=int); parser.add_argument("--gradient-accumulation-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for name in ("batch_size", "precision", "num_workers", "gradient_accumulation_steps"):
        if getattr(args, name) is not None: config[name] = getattr(args, name)
    if config.get("precision") not in PRECISIONS: raise ValueError("precision must be 32, 16-mixed, or bf16-mixed")
    ACTIVE_PRECISION = config["precision"]
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required for this baseline")
    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rows, integrity = load_rows(config)
    representation = config.get("target_representation", "exact_class")
    mapping = class_mapping_for_representation(representation) if representation == "ordinal_band" else {label: index for index, label in enumerate(sorted({row[config["target_column"]] for split_rows in rows.values() for row in split_rows}, key=class_sort_key))}
    labels = list(mapping)
    records, availability = attach_zarr_indices(rows, config, mapping)
    class_counts = {split: dict(sorted(Counter(record.target for record in split_records).items(), key=lambda item: mapping[item[0]])) for split, split_records in records.items()}
    metadata = {"config": config, "class_mapping": mapping, "class_counts": class_counts, "integrity": integrity, "image_availability": availability, "legacy_label_used_as_target": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "class_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    (output / "dataset_integrity.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "missing_or_unreadable_images.csv", availability["missing_or_unreadable"])
    write_csv(output / "class_distribution.csv", [{"split": split, "class": label, "count": count} for split, counts in class_counts.items() for label, count in counts.items()])
    datasets = {split: SuryaDataset(split_records, config["zarr_path"], availability["selected_channel_indices"]) for split, split_records in records.items()}
    loaders = {split: make_loader(dataset, config, shuffle=(split == "train")) for split, dataset in datasets.items()}
    device = torch.device("cuda")
    model = timm.create_model(config["model_name"], pretrained=bool(config["pretrained"]), in_chans=int(config["in_chans"]), num_classes=len(labels)).to(device)
    smoke = smoke_test(model, loaders["train"], config, device, output)
    print(f"smoke passed: {smoke}", flush=True)
    if args.smoke_only: return
    # Reinitialize after smoke: no smoke-test updates are included in the scientific run.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = timm.create_model(config["model_name"], pretrained=False, in_chans=int(config["in_chans"]), num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []; best_f1 = -math.inf; best_epoch = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        began = time.perf_counter(); train_loss, _, _ = run_epoch(model, loaders["train"], optimizer, scaler, device, int(config["gradient_accumulation_steps"]))
        validation_loss, val_true, val_pred = run_epoch(model, loaders["validation"], None, scaler, device, 1)
        validation_metrics, _, _ = metrics(val_true, val_pred, labels)
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, **{f"validation_{key}": value for key, value in validation_metrics.items()}, "elapsed_seconds": round(time.perf_counter() - began, 2)}
        history.append(row); write_csv(output / "training_log.csv", history)
        print(json.dumps(row), flush=True)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = validation_metrics["macro_f1"], epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer": "AdamW", "checkpoint_metric": "macro_f1", "validation_metrics": validation_metrics, "class_mapping": mapping}, output / "checkpoints" / "best_validation_macro_f1.pt")
    plot_curves(output / "learning_curves.svg", history)
    checkpoint = torch.load(output / "checkpoints" / "best_validation_macro_f1.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary_rows = []
    for split in ("validation", "test", "leaky_validation"):
        loss, true, pred = run_epoch(model, loaders[split], None, scaler, device, 1)
        summary, per_class, matrix = metrics(true, pred, labels); summary.update({"split": split, "loss": loss, "selected_epoch": best_epoch})
        summary_rows.append(summary); write_csv(output / f"{split}_per_class_metrics.csv", per_class)
        if split == "test":
            write_csv(output / "test_confusion_matrix.csv", [{"true_class": labels[i], **{labels[j]: int(matrix[i, j]) for j in range(len(labels))}} for i in range(len(labels))])
            plot_confusion(output / "test_confusion_matrix.svg", matrix, labels)
    write_csv(output / "summary_metrics.csv", summary_rows)
    (output / "checkpoint_metadata.json").write_text(json.dumps({"best_validation_epoch": best_epoch, "best_validation_macro_f1": best_f1, "selection_split": "validation", "test_not_used_for_selection": True, "smoke": smoke}, indent=2) + "\n", encoding="utf-8")
    print(f"completed best_epoch={best_epoch} best_validation_macro_f1={best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
