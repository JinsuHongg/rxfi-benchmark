"""Generate the ten regression target conditions from one-minute GOES XRS-B."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from scripts.data.generate_xrs_only_targets import (
    AnnualXrsCache,
    epoch_seconds,
    fetch_hf_splits,
    parse_timestamp,
    read_split,
    samples_in_window,
    selected_observations,
    sha256,
    write_csv,
)

SPLITS = ("train", "validation", "leaky_validation", "test")
DEFAULT_CONFIG = Path("configs/regression_target_experiment.json")
DEFAULT_HF_CACHE = Path("data/raw/surya_bench_splits_hf")
DEFAULT_XRS_CACHE = Path("data/raw/noaa_paper_background")
DEFAULT_INPUT_LOCK = Path("configs/paper_background_sensitivity.inputs.lock.json")
DEFAULT_DERIVED_DIR = Path("data/derived")
DEFAULT_OUTPUT_DIR = Path("outputs/regression_target_experiment")
def lower_tail_mean(values: np.ndarray, lower_percent: int) -> float | None:
    """Return the arithmetic mean of the requested lower-tail observations."""
    if not len(values):
        return None
    count = max(1, math.ceil(lower_percent * len(values) / 100))
    selected = np.partition(values, count - 1)[:count]
    return float(np.mean(selected, dtype=np.float64))


def target_columns(config: dict[str, Any]) -> list[str]:
    """Build the stable target column order from the experiment config."""
    columns = ["max_xflux", "fluence"]
    for condition in config["background_conditions"]:
        suffix = str(condition["id"])
        columns.extend((f"max_background_deviation_{suffix}", f"total_background_deviation_{suffix}"))
    return columns


def calculate_targets(
    prediction: np.ndarray,
    backgrounds: dict[str, float | None],
    sum_scale_seconds: int,
) -> dict[str, float | None]:
    """Calculate the shared flux targets and every signed deviation target."""
    targets: dict[str, float | None] = {
        "max_xflux": float(np.max(prediction)) if len(prediction) else None,
        "fluence": float(sum_scale_seconds * np.sum(prediction, dtype=np.float64)) if len(prediction) else None,
    }
    for condition_id, baseline in backgrounds.items():
        suffix = condition_id
        if baseline is None or not len(prediction):
            maximum = total = None
        else:
            deviations = prediction - baseline
            maximum = float(np.max(deviations))
            total = float(sum_scale_seconds * np.sum(deviations, dtype=np.float64))
        targets[f"max_background_deviation_{suffix}"] = maximum
        targets[f"total_background_deviation_{suffix}"] = total
    return targets


def generate_split(
    split: str,
    source_path: Path,
    staged_path: Path,
    timestamps: np.ndarray,
    flux: np.ndarray,
    satellites: np.ndarray,
    is_primary: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create one split CSV and its per-window coverage QC rows."""
    source_columns, source_rows = read_split(source_path, split)
    targets = target_columns(config)
    output_columns = [*source_columns, "original_row_index", *targets]
    generated_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for row_index, source in enumerate(source_rows):
        start = epoch_seconds(parse_timestamp(source["timestamp"]))
        end = start + int(config["prediction_window_hours"]) * 3600
        prediction, pred_sat, pred_primary = samples_in_window(timestamps, flux, satellites, is_primary, start, end)
        backgrounds: dict[str, float | None] = {}
        qc: dict[str, Any] = {
            "split": split,
            "original_row_index": row_index,
            "timestamp": source["timestamp"],
            "prediction_valid_count": len(prediction),
            "prediction_status": "ok" if len(prediction) else "no_valid_samples",
            "prediction_coverage_fraction": format(
                len(prediction) / (int(config["prediction_window_hours"]) * 60), ".8f"
            ),
            "prediction_primary_count": int(np.count_nonzero(pred_primary)),
            "prediction_secondary_count": int(len(pred_sat) - np.count_nonzero(pred_primary)),
        }
        for condition in config["background_conditions"]:
            condition_id = str(condition["id"])
            window_seconds = int(condition["window_hours"]) * 3600
            background_samples, bg_sat, bg_primary = samples_in_window(
                timestamps, flux, satellites, is_primary, start - window_seconds, start
            )
            baseline = lower_tail_mean(background_samples, int(condition["lower_percent"]))
            backgrounds[condition_id] = baseline
            qc[f"background_{condition_id}_valid_count"] = len(background_samples)
            qc[f"background_{condition_id}_status"] = "ok" if len(background_samples) else "no_valid_samples"
            qc[f"background_{condition_id}_coverage_fraction"] = format(
                len(background_samples) / (window_seconds // 60), ".8f"
            )
            qc[f"background_{condition_id}_primary_count"] = int(np.count_nonzero(bg_primary))
            qc[f"background_{condition_id}_secondary_count"] = int(len(bg_sat) - np.count_nonzero(bg_primary))
            qc[f"background_{condition_id}_mean"] = "" if baseline is None else format(baseline, ".12g")
        generated = dict(source)
        generated["original_row_index"] = str(row_index)
        for name, value in calculate_targets(
            prediction, backgrounds, int(config["sample_sum_scale_seconds"])
        ).items():
            generated[name] = "" if value is None else format(value, ".12g")
        generated_rows.append(generated)
        qc_rows.append(qc)
    write_csv(staged_path, output_columns, generated_rows)
    validate_alignment(source_rows, generated_rows, source_columns, split)
    summary = {
        "rows": len(source_rows),
        "source_sha256": sha256(source_path),
        "target_sha256": sha256(staged_path),
        "valid_counts": {name: sum(row[name] != "" for row in generated_rows) for name in targets},
    }
    return qc_rows, summary


def validate_alignment(
    source_rows: list[dict[str, str]],
    generated_rows: list[dict[str, Any]],
    source_columns: list[str],
    split: str,
) -> None:
    """Ensure target generation preserves every source value and row order."""
    if len(source_rows) != len(generated_rows):
        raise RuntimeError(f"row count changed for {split}: {len(source_rows)} != {len(generated_rows)}")
    for row_index, (source, generated) in enumerate(zip(source_rows, generated_rows)):
        if int(generated["original_row_index"]) != row_index:
            raise RuntimeError(f"row index changed for {split} at row {row_index}")
        for column in source_columns:
            if generated[column] != source[column]:
                raise RuntimeError(f"source column {column} changed for {split} at row {row_index}")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen target-generation configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    conditions_raw = config.get("background_conditions")
    if not isinstance(conditions_raw, list) or len(cast(list[Any], conditions_raw)) != 4:
        raise ValueError("experiment config must define exactly four background conditions")
    conditions = cast(list[dict[str, Any]], conditions_raw)
    condition_ids = [str(item["id"]) for item in conditions]
    expected_ids = {"2h_q10", "2h_q50", "24h_q10", "24h_q50"}
    if set(condition_ids) != expected_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("background condition IDs must be 2h_q10, 2h_q50, 24h_q10, and 24h_q50")
    expected = {
        "2h_q10": (2, 10),
        "2h_q50": (2, 50),
        "24h_q10": (24, 10),
        "24h_q50": (24, 50),
    }
    for item in conditions:
        actual = (int(item["window_hours"]), int(item["lower_percent"]))
        if actual != expected[str(item["id"])]:
            raise ValueError(f"background condition does not match its ID: {item}")
    if int(config["prediction_window_hours"]) != 24 or int(config["sample_sum_scale_seconds"]) != 1:
        raise ValueError("the experiment requires a 24-hour horizon and unscaled one-minute sums")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible split generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default="main", help="Hugging Face revision or full commit SHA")
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--xrs-cache-dir", type=Path, default=DEFAULT_XRS_CACHE)
    parser.add_argument("--xrs-input-lock", type=Path, default=DEFAULT_INPUT_LOCK)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="use cached HF and NOAA inputs only")
    parser.add_argument("--overwrite", action="store_true", help="replace existing train/validation/test CSVs")
    return parser.parse_args()


def main() -> None:
    """Generate all four split files plus checksummed provenance and QC."""
    args = parse_args()
    config = load_config(args.config)
    final_paths = {split: args.derived_dir / f"{split}.csv" for split in SPLITS}
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output exists; pass --overwrite to replace: " + ", ".join(map(str, existing)))

    split_paths, hf_inputs = fetch_hf_splits(args.revision, args.hf_cache_dir, offline=args.offline)
    split_data = {split: read_split(split_paths[split], split) for split in SPLITS}
    all_datetimes = [parse_timestamp(row["timestamp"]) for _, rows in split_data.values() for row in rows]
    max_background = max(int(row["window_hours"]) for row in config["background_conditions"])
    span_start = epoch_seconds(min(all_datetimes) - timedelta(hours=max_background))
    span_end = epoch_seconds(max(all_datetimes) + timedelta(hours=int(config["prediction_window_hours"])))
    cache = AnnualXrsCache(args.xrs_cache_dir, args.xrs_input_lock, offline=args.offline)
    times, flux, satellites, is_primary, missing_samples = selected_observations(span_start, span_end, cache)

    run_manifest = {
        "config": config,
        "satellite_sample_selection": "flag-zero primary, then flag-zero secondary; record uncovered minutes; retain split rows and leave affected target values blank when their required window has no valid samples",
        "hugging_face": hf_inputs,
        "noaa": cache.input_manifest,
    }
    run_id = hashlib.sha256(json.dumps(run_manifest, sort_keys=True).encode()).hexdigest()[:16]
    staging = args.derived_dir / f".regression_target_staging_{run_id}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    qc_all: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        qc_rows, summary = generate_split(
            split, split_paths[split], staging / f"{split}.csv", times, flux, satellites, is_primary, config
        )
        qc_all.extend(qc_rows)
        split_summaries[split] = summary

    args.derived_dir.mkdir(parents=True, exist_ok=True)
    for split, final_path in final_paths.items():
        staging.joinpath(f"{split}.csv").replace(final_path)
    staging.rmdir()

    run_dir = args.output_dir / "runs" / run_id
    qc_path = run_dir / "target_coverage_qc.csv"
    write_csv(qc_path, list(qc_all[0]), qc_all)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "satellite_sample_selection": run_manifest["satellite_sample_selection"],
        "hugging_face": hf_inputs,
        "noaa_files": list(cache.input_manifest.values()),
        "unavailable_noaa_files": cache.unavailable,
        "missing_flag_zero_samples": {
            "count": len(missing_samples),
            "path": str(run_dir / "missing_xrs_samples.csv"),
            "policy": "skip minutes without a flag-zero sample from either scheduled satellite; list each skipped minute for QC",
        },
        "empty_target_windows": "retain every split row and source timestamp; leave target values blank when a required prediction or background window has no valid samples",
        "split_outputs": {split: {"path": str(path), **split_summaries[split]} for split, path in final_paths.items()},
        "target_coverage_qc": {"path": str(qc_path), "sha256": sha256(qc_path)},
    }
    missing_path = run_dir / "missing_xrs_samples.csv"
    write_csv(
        missing_path,
        ["timestamp_utc", "scheduled_primary_satellite", "scheduled_secondary_satellite", "primary_status", "secondary_status"],
        missing_samples,
    )
    metadata_path = run_dir / "target_generation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Generated regression target splits: {}", ", ".join(map(str, final_paths.values())))
    logger.info("Run metadata: {}", metadata_path)


if __name__ == "__main__":
    main()
