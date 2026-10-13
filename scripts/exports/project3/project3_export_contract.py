#!/usr/bin/env python3
"""Standalone file contract for Project 3 prediction exports.

This module deliberately depends only on the Python standard library so the
consumer can copy it into ``ordinal-cqr`` without importing this repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "project3_prediction_export_v1"
DATASET = "surya_bench_noaa_ncei_24h"
SOURCE_COLUMNS = (
    "split", "original_row_index", "timestamp", "target_name", "target_raw",
    "target_transformed", "max_flare_class", "q05", "q50", "q95", "q05_raw",
    "q50_raw", "q95_raw",
)
EXPORT_COLUMNS = (
    "dataset", "experiment_id", "split", "original_row_index", "timestamp",
    "target_name", "target_raw", "target_transformed", "max_flare_class",
    "max_flare_class_raw", "ordinal_class", "ordinal_class_index", "q05", "q50",
    "q95", "q05_raw", "q50_raw", "q95_raw",
)
CLASS_ORDER = ("FQ", "A", "B", "C", "M", "X")
CLASS_MAPPING = {name: index for index, name in enumerate(CLASS_ORDER)}
PHYSICAL_SPLITS = ("validation", "leaky_validation", "test")
SPLIT_ROLE_MAPPING = {
    "train": {"logical_role": "predictive_model_fitting"},
    "validation": {"logical_role": "checkpoint_selection"},
    "leaky_validation": {
        "source_split": "leaky_validation",
        "logical_role": "calibration",
        "proposal_status": "preliminary_integration_plan",
    },
    "test": {"logical_role": "final_conformal_evaluation"},
}
PRIORITY_TARGETS = {"max_peak_flux", "cumulative_peak_flux"}
NOAA_LABEL = re.compile(r"([ABCMX])(?:[0-9]+(?:\.[0-9]+)?)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ordinal_class(label: str) -> tuple[str, int]:
    """Return a canonical broad band and fixed ordinal index for a NOAA label."""
    normalized = str(label).strip().upper()
    if normalized == "FQ":
        return "FQ", CLASS_MAPPING["FQ"]
    match = NOAA_LABEL.fullmatch(normalized)
    if not match:
        raise ValueError(f"Invalid NOAA flare label: {label!r}")
    ordinal = match.group(1)
    return ordinal, CLASS_MAPPING[ordinal]


def validate_transform(spec: dict[str, Any]) -> dict[str, Any]:
    name = spec.get("name") if isinstance(spec, dict) else None
    if name not in {"log10_scaled", "log10_1p_scaled", "log10_1p"}:
        raise ValueError(f"Unsupported target transform: {name!r}")
    result = {"name": name}
    if name != "log10_1p":
        scale = float(spec.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Scaled target transforms require a positive finite scale")
        result["scale"] = scale
    return result


def forward_transform(raw: float, spec: dict[str, Any]) -> float:
    spec = validate_transform(spec)
    raw = float(raw)
    if not math.isfinite(raw):
        raise ValueError("Raw target must be finite")
    if spec["name"] == "log10_scaled":
        if raw <= 0:
            raise ValueError("log10_scaled requires a positive raw target")
        return math.log10(raw / spec["scale"])
    if raw < 0:
        raise ValueError(f"{spec['name']} requires a non-negative raw target")
    if spec["name"] == "log10_1p_scaled":
        return math.log10(1 + raw / spec["scale"])
    return math.log10(1 + raw)


def inverse_transform(value: float, spec: dict[str, Any]) -> float:
    spec = validate_transform(spec)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Transformed value must be finite")
    if spec["name"] == "log10_scaled":
        return spec["scale"] * 10 ** value
    if spec["name"] == "log10_1p_scaled":
        return spec["scale"] * (10 ** value - 1)
    return 10 ** value - 1


def stable_row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], str(row["original_row_index"]), row["timestamp"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_columns(fieldnames: Iterable[str] | None, expected: Iterable[str], context: str) -> None:
    found = tuple(fieldnames or ())
    missing = set(expected) - set(found)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")


def source_experiment_dir(root: Path, target: str) -> Path:
    path = root / "outputs" / f"vit_small_224_{target}_qr"
    if not path.is_dir():
        raise FileNotFoundError(f"Completed QR experiment not found: {path}")
    return path


def build_metadata(experiment_dir: Path, target: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    experiment = read_json(experiment_dir / "experiment_metadata.json")
    transform = validate_transform(read_json(experiment_dir / "transform_metadata.json"))
    checkpoint = read_json(experiment_dir / "checkpoint_metadata.json")
    config = experiment["config"]
    selected_channels = config.get("selected_channels", config["channel_order"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "experiment_id": config["experiment_name"],
        "target_name": target,
        "target_transform": transform["name"],
        "target_transform_parameters": {key: value for key, value in transform.items() if key != "name"},
        "quantiles": config["quantiles"],
        "forecast_horizon_hours": 24,
        "model": {"name": config["model_name"], "patch_size": config.get("patch_size"), "pretrained": config.get("pretrained")},
        "source_model": config["model_name"],
        "image_configuration": {
            "image_size": config["image_size"], "in_chans": config["in_chans"],
            "selected_channels": selected_channels, "normalization": config["normalization"],
        },
        "seed": config["seed"],
        "checkpoint_epoch": checkpoint["best_validation_epoch"],
        "checkpoint_selection_metric": config["checkpoint_metric"],
        "checkpoint_selection_split": checkpoint["selection_split"],
        "source_prediction_directory": str((experiment_dir / "predictions").resolve()),
        "source_prediction_sha256": source_hashes,
        "source_split_integrity": experiment.get("integrity", {}).get("source_split_sha256", {}),
        "derived_target_integrity": experiment.get("integrity", {}).get("derived_target_sha256", {}),
        "class_order": list(CLASS_ORDER), "class_mapping": CLASS_MAPPING,
        "split_role_mapping": SPLIT_ROLE_MAPPING,
        "git_commit": experiment.get("git_commit"),
        "created_at": datetime.now().astimezone().isoformat(),
        "target_priority": "primary" if target in PRIORITY_TARGETS else "secondary",
        "source_predictions": "completed_best_validation_checkpoint_predictions",
        "scope": "prediction_export_only_no_conformal_methodology",
    }


def export_target(experiment_dir: Path, output_root: Path, target: str) -> Path:
    """Materialize one canonical export from completed QR prediction CSVs."""
    experiment_dir, output_root = Path(experiment_dir), Path(output_root)
    output_dir = output_root / target
    source_hashes: dict[str, str] = {}
    for split in PHYSICAL_SPLITS:
        source = experiment_dir / "predictions" / f"{split}_predictions.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing completed prediction export: {source}")
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader.fieldnames, SOURCE_COLUMNS, str(source))
            rows: list[dict[str, Any]] = []
            for row in reader:
                if row["split"] != split:
                    raise ValueError(f"{source} contains unexpected split {row['split']!r}")
                ordinal, ordinal_index = parse_ordinal_class(row["max_flare_class"])
                rows.append({
                    "dataset": DATASET, "experiment_id": experiment_dir.name,
                    **{name: row[name] for name in SOURCE_COLUMNS},
                    "max_flare_class_raw": row["max_flare_class"],
                    "ordinal_class": ordinal, "ordinal_class_index": ordinal_index,
                })
        write_csv(output_dir / f"{split}.csv", rows)
        source_hashes[split] = sha256(source)
    metadata = build_metadata(experiment_dir, target, source_hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def _finite(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: {field} must be finite")
    return value


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10)


def validate_export_directory(export_dir: Path) -> dict[str, Any]:
    export_dir = Path(export_dir)
    metadata = read_json(export_dir / "metadata.json")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {metadata.get('schema_version')!r}")
    transform = {"name": metadata.get("target_transform"), **metadata.get("target_transform_parameters", {})}
    transform = validate_transform(transform)
    roles = metadata.get("split_role_mapping")
    if roles != SPLIT_ROLE_MAPPING:
        raise ValueError("split_role_mapping does not match the v1 physical-split contract")
    seen: set[tuple[str, str, str]] = set()
    counts: dict[str, int] = {}
    crossings = {"q05_gt_q50": 0, "q50_gt_q95": 0, "q05_gt_q95": 0}
    for split in PHYSICAL_SPLITS:
        path = export_dir / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader.fieldnames, EXPORT_COLUMNS, str(path))
            local: set[tuple[str, str, str]] = set()
            count = 0
            for row in reader:
                context = f"{path}:{count + 2}"
                if row["dataset"] != metadata["dataset"] or row["experiment_id"] != metadata["experiment_id"]:
                    raise ValueError(f"{context}: dataset or experiment_id disagrees with metadata")
                if row["split"] != split:
                    raise ValueError(f"{context}: split must be {split!r}")
                try:
                    datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    int(row["original_row_index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{context}: invalid timestamp or original_row_index") from exc
                key = stable_row_key(row)
                if key in local or key in seen:
                    raise ValueError(f"{context}: duplicate or cross-split row identity {key!r}")
                local.add(key); seen.add(key)
                ordinal, ordinal_index = parse_ordinal_class(row["max_flare_class_raw"])
                if row["max_flare_class"] != row["max_flare_class_raw"] or row["ordinal_class"] != ordinal or int(row["ordinal_class_index"]) != ordinal_index:
                    raise ValueError(f"{context}: invalid canonical class representation")
                raw = _finite(row, "target_raw", context)
                transformed = _finite(row, "target_transformed", context)
                if not _close(forward_transform(raw, transform), transformed):
                    raise ValueError(f"{context}: target transform is inconsistent")
                quantiles = {name: _finite(row, name, context) for name in ("q05", "q50", "q95")}
                for name, transformed_prediction in quantiles.items():
                    raw_name = f"{name}_raw"
                    if not _close(inverse_transform(transformed_prediction, transform), _finite(row, raw_name, context)):
                        raise ValueError(f"{context}: {raw_name} inverse transform is inconsistent")
                crossings["q05_gt_q50"] += quantiles["q05"] > quantiles["q50"]
                crossings["q50_gt_q95"] += quantiles["q50"] > quantiles["q95"]
                crossings["q05_gt_q95"] += quantiles["q05"] > quantiles["q95"]
                count += 1
        counts[split] = count
        source = Path(metadata["source_prediction_directory"]) / f"{split}_predictions.csv"
        expected_hash = metadata.get("source_prediction_sha256", {}).get(split)
        if source.is_file() and expected_hash and sha256(source) != expected_hash:
            raise ValueError(f"Source prediction changed after export: {source}")
    return {"target": metadata["target_name"], "rows": counts, "quantile_crossings": crossings}
