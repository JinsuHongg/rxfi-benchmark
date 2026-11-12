"""Evaluate regression-derived binary flare skill from saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import TypeAlias, cast

from loguru import logger

from scripts.analysis.severe_event_metrics import (
    binary_metrics_from_counts,
    ranking_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET_NAMES = (
    "fluence",
    "max_background_deviation_24h_q10",
    "max_background_deviation_24h_q50",
    "max_background_deviation_2h_q10",
    "max_background_deviation_2h_q50",
    "max_xflux",
    "total_background_deviation_24h_q10",
    "total_background_deviation_24h_q50",
    "total_background_deviation_2h_q10",
    "total_background_deviation_2h_q50",
)
CHANNEL_CONDITIONS = (
    "all13",
    "aia8",
    "hmi5",
    "single_aia131",
    "single_aia171",
    "single_aia193",
    "single_aia211",
    "single_aia1600",
    "single_hmi_m",
)
METRIC_NAMES = (
    "n",
    "n_positive",
    "n_negative",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision_positive",
    "POD",
    "FAR",
    "FPR",
    "FNR",
    "specificity",
    "F1_positive",
    "F1_negative",
    "F1_macro",
    "balanced_accuracy",
    "TSS",
    "HSS",
    "CSS",
    "frequency_bias",
    "positive_prevalence",
    "prediction_rate",
    "AUROC",
    "AUPRC",
)
CURVE_METRIC_NAMES = (
    "TP",
    "FP",
    "TN",
    "FN",
    "POD",
    "FAR",
    "F1_macro",
    "TSS",
    "HSS",
    "CSS",
)
KEY_COLUMNS = ("timestamp", "original_row_index")
CsvValue: TypeAlias = str | int | float | bool
CsvRecord: TypeAlias = dict[str, CsvValue]


@dataclass(frozen=True)
class RunSpec:
    """Identify one selected checkpoint and its saved split predictions."""

    family: str
    target: str
    condition: str
    seed: int
    checkpoint_path: str
    validation_prediction: Path
    test_prediction: Path


@dataclass(frozen=True)
class EvaluationArgs:
    """Typed command-line options for the evaluation."""

    root: Path
    split_dir: Path
    target_artifacts_dir: Path
    channel_test_summary: Path
    output_dir: Path
    max_xflux_threshold: float


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV as string-valued records."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [cast(dict[str, str], row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[CsvRecord], columns: list[str]) -> None:
    """Write records using a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        _ = cast(None, writer.writeheader())
        writer.writerows(rows)


def canonical_path(root: Path, value: str) -> str:
    """Normalize absolute and repository-relative artifact paths."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def prediction_index(root: Path, artifact_dir: Path) -> dict[tuple[str, str], Path]:
    """Map a saved checkpoint and split to its prediction CSV."""
    index: dict[tuple[str, str], Path] = {}
    for metric_file in sorted((artifact_dir / "metrics").glob("*_metrics.csv")):
        for row in read_csv(metric_file):
            checkpoint = canonical_path(root, row["checkpoint_path"])
            prediction = artifact_dir / "predictions" / row["prediction_file"]
            if not prediction.is_file():
                continue
            split = prediction_split(row["prediction_file"])
            if split is None:
                continue
            key = (checkpoint, split)
            previous = index.get(key)
            if previous is not None and previous != prediction:
                _ = logger.warning(
                    "Repeated {} inference for one checkpoint; replacing {} with {}",
                    split,
                    previous,
                    prediction,
                )
            index[key] = prediction
    return index


def prediction_split(filename: str) -> str | None:
    """Infer split from the established prediction filename suffix."""
    for split in ("validation", "test"):
        if filename.endswith(f"_{split}_predictions.csv"):
            return split
    return None


def resolve_prediction(
    root: Path,
    index: dict[tuple[str, str], Path],
    checkpoint_path: str,
    split: str,
) -> Path:
    """Find a selected checkpoint's saved prediction file."""
    key = (canonical_path(root, checkpoint_path), split)
    if key not in index:
        raise FileNotFoundError(f"no saved {split} prediction for checkpoint {key[0]}")
    return index[key]


def target_specs(
    root: Path, artifacts: Path, index: dict[tuple[str, str], Path]
) -> list[RunSpec]:
    """Build paired validation/test records for selected target checkpoints."""
    manifest_dir = artifacts / "manifests"
    validation_rows = read_csv(manifest_dir / "validation.csv")
    test_rows = read_csv(manifest_dir / "targets_test.csv")
    validation = {(row["target"], int(row["seed"])): row for row in validation_rows}
    tests = {(row["target"], int(row["seed"])): row for row in test_rows}
    expected = {(target, seed) for target in TARGET_NAMES for seed in (1, 2, 3)}
    if set(validation) != expected or set(tests) != expected:
        raise ValueError("target manifests must contain all 10 targets for seeds 1–3")

    specs: list[RunSpec] = []
    for target, seed in sorted(expected):
        val_checkpoint = validation[(target, seed)]["checkpoint_path"]
        test_checkpoint = tests[(target, seed)]["checkpoint_path"]
        if canonical_path(root, val_checkpoint) != canonical_path(
            root, test_checkpoint
        ):
            raise ValueError(
                f"validation/test checkpoint mismatch for {target} seed {seed}"
            )
        specs.append(
            RunSpec(
                family="target",
                target=target,
                condition="all13",
                seed=seed,
                checkpoint_path=val_checkpoint,
                validation_prediction=resolve_prediction(
                    root, index, val_checkpoint, "validation"
                ),
                test_prediction=resolve_prediction(
                    root, index, test_checkpoint, "test"
                ),
            )
        )
    return specs


def channel_specs(
    root: Path, summary_path: Path, index: dict[tuple[str, str], Path]
) -> list[RunSpec]:
    """Build paired predictions for the nine established channel conditions."""
    rows = read_csv(summary_path)
    selected = [row for row in rows if row["input_condition"] in CHANNEL_CONDITIONS]
    actual = {(row["input_condition"], int(row["seed"])) for row in selected}
    expected = {
        (condition, seed) for condition in CHANNEL_CONDITIONS for seed in (1, 2, 3)
    }
    if actual != expected or len(selected) != len(expected):
        raise ValueError(
            "channel test summary must contain nine conditions for seeds 1–3"
        )

    specs: list[RunSpec] = []
    for row in selected:
        checkpoint = row["checkpoint_path"]
        test_prediction = find_channel_test_prediction(root, summary_path, row)
        specs.append(
            RunSpec(
                family="channel_ablation",
                target="fluence",
                condition=row["input_condition"],
                seed=int(row["seed"]),
                checkpoint_path=checkpoint,
                validation_prediction=resolve_prediction(
                    root, index, checkpoint, "validation"
                ),
                test_prediction=test_prediction,
            )
        )
    return sorted(specs, key=lambda spec: (spec.condition, spec.seed))


def find_channel_test_prediction(
    root: Path, summary_path: Path, row: dict[str, str]
) -> Path:
    """Resolve channel test predictions across the two historical output roots."""
    candidates = (
        summary_path.parent.parent / "predictions" / row["prediction_file"],
        root
        / "outputs/xrs_only/v1_test_evaluation/predictions"
        / row["prediction_file"],
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing channel test prediction: {row['prediction_file']}"
    )


def flare_binary(value: str) -> int:
    """Map NOAA flare classes to M1.0-or-higher labels."""
    flare_class = value.strip().upper()
    if not flare_class:
        raise ValueError("empty max_goes_class")
    if flare_class.startswith("X"):
        magnitude = flare_class[1:]
        return int(not magnitude or float(magnitude) > 0.0)
    if flare_class.startswith("M"):
        magnitude = flare_class[1:]
        if not magnitude:
            raise ValueError(f"M-class label is missing its magnitude: {value!r}")
        return int(float(magnitude) >= 1.0)
    return 0


def sample_key(row: dict[str, str]) -> tuple[str, int]:
    """Return the stable timestamp and row-index join key."""
    return (row["timestamp"], int(row["original_row_index"]))


def split_labels(path: Path) -> dict[tuple[str, int], tuple[int, str]]:
    """Load and validate stored binary labels against NOAA flare classes."""
    labels: dict[tuple[str, int], tuple[int, str]] = {}
    for row in read_csv(path):
        key = sample_key(row)
        truth = flare_binary(row["max_goes_class"])
        stored = int(row["label_max"])
        if truth != stored:
            raise ValueError(f"label_max disagrees with max_goes_class at {key}")
        if key in labels:
            raise ValueError(f"duplicate split sample key: {key}")
        labels[key] = (truth, row["max_goes_class"])
    return labels


def load_scored_samples(
    path: Path,
    expected_split: str,
    expected_target: str,
    labels: dict[tuple[str, int], tuple[int, str]],
) -> tuple[list[int], list[float], int]:
    """Join saved q50 predictions with the authoritative split labels."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    required = {"split", "target_name", "max_flare_class", "q50_raw", *KEY_COLUMNS}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path} missing columns {sorted(required - set(rows[0]))}")
    truths: list[int] = []
    scores: list[float] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row["split"] != expected_split or row["target_name"] != expected_target:
            raise ValueError(f"split/target mismatch in {path}")
        key = sample_key(row)
        if key not in labels:
            raise ValueError(f"prediction sample absent from split file: {key}")
        if key in seen:
            raise ValueError(f"duplicate prediction sample key in {path}: {key}")
        seen.add(key)
        truth, _ = labels[key]
        if flare_binary(row["max_flare_class"]) != truth:
            raise ValueError(f"prediction flare label mismatch at {key} in {path}")
        score = float(row["q50_raw"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite q50_raw at {key} in {path}")
        truths.append(truth)
        scores.append(score)
    return truths, scores, len(labels) - len(seen)


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return NaN when the requested contingency denominator is absent."""
    return float(numerator / denominator) if denominator else float("nan")


def calculate_css(tss: float, hss: float) -> float:
    """Return zero for negative TSS/HSS and their geometric mean otherwise."""
    if not math.isfinite(tss) or not math.isfinite(hss):
        return float("nan")
    if tss < 0 or hss < 0:
        return 0.0
    return math.sqrt(tss * hss)


def evaluate_scores(
    y_true: list[int], y_pred: list[int], scores: list[float]
) -> dict[str, float | int]:
    """Calculate requested operational metrics and supporting counts."""
    counts = list(zip(y_true, y_pred, strict=True))
    tp = sum(truth == 1 and predicted == 1 for truth, predicted in counts)
    fp = sum(truth == 0 and predicted == 1 for truth, predicted in counts)
    tn = sum(truth == 0 and predicted == 0 for truth, predicted in counts)
    fn = sum(truth == 1 and predicted == 0 for truth, predicted in counts)
    result = binary_metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)
    tss, hss = float(result["TSS"]), float(result["HSS"])
    result.update(
        {
            "FAR": safe_ratio(fp, tp + fp),
            "CSS": calculate_css(tss, hss),
            "frequency_bias": safe_ratio(tp + fp, tp + fn),
            "positive_prevalence": safe_ratio(tp + fn, tp + fp + tn + fn),
            "prediction_rate": safe_ratio(tp + fp, tp + fp + tn + fn),
            **ranking_metrics(y_true, scores),
        }
    )
    return result


def threshold_candidates(
    y_true: list[int], scores: list[float]
) -> list[dict[str, float | int]]:
    """Evaluate each observed score and an all-negative decision threshold."""
    ranked = sorted(
        zip(scores, y_true, strict=True), key=lambda pair: pair[0], reverse=True
    )
    positives = sum(y_true)
    negatives = len(y_true) - positives
    candidates: list[dict[str, float | int]] = []
    ranking = ranking_metrics(y_true, scores)
    tp = fp = 0
    cursor = 0
    while cursor < len(ranked):
        threshold = ranked[cursor][0]
        end = cursor
        while end < len(ranked) and ranked[end][0] == threshold:
            if ranked[end][1] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        tn, fn = negatives - fp, positives - tp
        candidates.append(
            {
                "threshold": threshold,
                **metrics_from_counts(tp, fp, tn, fn, positives, len(y_true), ranking),
            }
        )
        cursor = end
    above_max = math.nextafter(ranked[0][0], math.inf)
    candidates.append(
        {
            "threshold": above_max,
            **metrics_from_counts(
                0, 0, negatives, positives, positives, len(y_true), ranking
            ),
        }
    )
    return candidates


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    positives: int,
    sample_count: int,
    ranking: dict[str, float],
) -> dict[str, float | int]:
    """Calculate classification metrics from counts and cached ranks."""
    result = binary_metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)
    tss, hss = float(result["TSS"]), float(result["HSS"])
    result.update(
        {
            "FAR": safe_ratio(fp, tp + fp),
            "CSS": calculate_css(tss, hss),
            "frequency_bias": safe_ratio(tp + fp, positives),
            "positive_prevalence": safe_ratio(positives, sample_count),
            "prediction_rate": safe_ratio(tp + fp, sample_count),
            **ranking,
        }
    )
    return result


def sort_value(row: dict[str, float | int], key: str) -> float:
    """Convert undefined tie-break metrics to negative infinity."""
    value = float(row[key])
    return value if math.isfinite(value) else float("-inf")


def select_threshold(
    y_true: list[int], scores: list[float]
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Select validation threshold by CSS and documented tie breakers."""
    curve = threshold_candidates(y_true, scores)
    selected = max(
        curve,
        key=lambda row: (
            sort_value(row, "CSS"),
            sort_value(row, "TSS"),
            sort_value(row, "HSS"),
            sort_value(row, "POD"),
            float(row["threshold"]),
        ),
    )
    return selected, curve


def evaluate_run(
    root: Path,
    spec: RunSpec,
    split: str,
    path: Path,
    threshold: float,
    labels: dict[tuple[str, int], tuple[int, str]],
) -> tuple[CsvRecord, int]:
    """Evaluate one saved prediction file with a fixed threshold."""
    truth, scores, unmatched = load_scored_samples(path, split, spec.target, labels)
    prediction = [int(score >= threshold) for score in scores]
    metrics = evaluate_scores(truth, prediction, scores)
    row: CsvRecord = {
        "family": spec.family,
        "target": spec.target,
        "condition": spec.condition,
        "seed": spec.seed,
        "split": split,
        "threshold": threshold,
        "threshold_selection": "validation_max_CSS",
        "n_unmatched_split_rows": unmatched,
        "prediction_file": str(path.relative_to(root)),
        **metrics,
    }
    return row, unmatched


def evaluate_physical_max_xflux(
    root: Path,
    spec: RunSpec,
    split: str,
    path: Path,
    labels: dict[tuple[str, int], tuple[int, str]],
    threshold: float,
) -> CsvRecord:
    """Evaluate the M1.0 physical flux threshold for max_xflux runs."""
    truth, scores, unmatched = load_scored_samples(path, split, spec.target, labels)
    prediction = [int(score >= threshold) for score in scores]
    metrics = evaluate_scores(truth, prediction, scores)
    return {
        "family": spec.family,
        "target": spec.target,
        "condition": spec.condition,
        "seed": spec.seed,
        "split": split,
        "threshold": threshold,
        "threshold_rule": "q50_raw >= M1.0 flux threshold",
        "n_unmatched_split_rows": unmatched,
        "prediction_file": str(path.relative_to(root)),
        **metrics,
    }


def aggregate_metrics(rows: list[CsvRecord]) -> list[CsvRecord]:
    """Summarize each condition and split over available random seeds."""
    groups: dict[tuple[str, str, str, str], list[CsvRecord]] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["target"]),
            str(row["condition"]),
            str(row["split"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: list[CsvRecord] = []
    for key, group in sorted(groups.items()):
        summary: CsvRecord = dict(zip(("family", "target", "condition", "split"), key))
        summary["seed_count"] = len(group)
        for metric in METRIC_NAMES:
            values = [
                float(row[metric]) for row in group if math.isfinite(float(row[metric]))
            ]
            summary[f"{metric}_mean"] = mean(values) if values else float("nan")
            summary[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
            )
        summaries.append(summary)
    return summaries


def parse_args() -> EvaluationArgs:
    """Parse repository and output locations from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, default=ROOT)
    _ = parser.add_argument(
        "--split-dir", type=Path, default=ROOT / "data/derived_noaa_g17_v222"
    )
    _ = parser.add_argument(
        "--target-artifacts-dir",
        type=Path,
        default=ROOT / "outputs/xrs_only/v1_evaluation_reselection",
    )
    _ = parser.add_argument(
        "--channel-test-summary",
        type=Path,
        default=ROOT
        / "outputs/xrs_only/v1_channel_ablation_test/summary/test_metrics_summary.csv",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/xrs_only/v1_operational_skill",
    )
    _ = parser.add_argument("--max-xflux-threshold", type=float, default=1e-5)
    parsed = parser.parse_args()
    return EvaluationArgs(
        root=cast(Path, parsed.root),
        split_dir=cast(Path, parsed.split_dir),
        target_artifacts_dir=cast(Path, parsed.target_artifacts_dir),
        channel_test_summary=cast(Path, parsed.channel_test_summary),
        output_dir=cast(Path, parsed.output_dir),
        max_xflux_threshold=cast(float, parsed.max_xflux_threshold),
    )


def output_columns() -> list[str]:
    """Return the stable columns for run-level and aggregate tables."""
    identifiers = [
        "family",
        "target",
        "condition",
        "seed",
        "split",
        "threshold",
        "threshold_selection",
        "n_unmatched_split_rows",
        "prediction_file",
    ]
    return identifiers + list(METRIC_NAMES)


def evaluate_spec(
    root: Path,
    spec: RunSpec,
    labels: dict[str, dict[tuple[str, int], tuple[int, str]]],
    physical_threshold: float,
) -> tuple[list[CsvRecord], CsvRecord, list[CsvRecord], list[CsvRecord]]:
    """Evaluate one selected checkpoint and create its threshold trace."""
    truth, scores, _ = load_scored_samples(
        spec.validation_prediction, "validation", spec.target, labels["validation"]
    )
    selected, curve = select_threshold(truth, scores)
    threshold = float(selected["threshold"])
    thresholds = make_threshold_record(root, spec, selected, threshold)
    curve_rows = make_curve_records(spec, curve, threshold)
    metrics, physical = evaluate_spec_splits(
        root, spec, labels, threshold, physical_threshold
    )
    return metrics, thresholds, curve_rows, physical


def make_threshold_record(
    root: Path, spec: RunSpec, selected: dict[str, float | int], threshold: float
) -> CsvRecord:
    """Describe a selected validation threshold and its paired prediction files."""
    return {
        "family": spec.family,
        "target": spec.target,
        "condition": spec.condition,
        "seed": spec.seed,
        "selected_threshold": threshold,
        "validation_CSS": selected["CSS"],
        "validation_TSS": selected["TSS"],
        "validation_HSS": selected["HSS"],
        "validation_POD": selected["POD"],
        "validation_prediction_file": str(spec.validation_prediction.relative_to(root)),
        "test_prediction_file": str(spec.test_prediction.relative_to(root)),
        "checkpoint_path": spec.checkpoint_path,
    }


def make_curve_records(
    spec: RunSpec, curve: list[dict[str, float | int]], selected: float
) -> list[CsvRecord]:
    """Attach run identity and selected marker to each validation threshold."""
    return [
        {
            "family": spec.family,
            "target": spec.target,
            "condition": spec.condition,
            "seed": spec.seed,
            "threshold": candidate["threshold"],
            **{name: candidate[name] for name in CURVE_METRIC_NAMES},
            "selected": float(candidate["threshold"]) == selected,
        }
        for candidate in curve
    ]


def evaluate_spec_splits(
    root: Path,
    spec: RunSpec,
    labels: dict[str, dict[tuple[str, int], tuple[int, str]]],
    threshold: float,
    physical_threshold: float,
) -> tuple[list[CsvRecord], list[CsvRecord]]:
    """Evaluate validation and test predictions at fixed thresholds."""
    metrics: list[CsvRecord] = []
    physical: list[CsvRecord] = []
    predictions = (
        ("validation", spec.validation_prediction),
        ("test", spec.test_prediction),
    )
    for split, path in predictions:
        result, _ = evaluate_run(root, spec, split, path, threshold, labels[split])
        metrics.append(result)
        if spec.target == "max_xflux":
            physical.append(
                evaluate_physical_max_xflux(
                    root, spec, split, path, labels[split], physical_threshold
                )
            )
    return metrics, physical


def write_metric_tables(output_dir: Path, rows: list[CsvRecord]) -> None:
    """Write per-run and seed-aggregated binary metric tables."""
    write_csv(output_dir / "per_run_metrics.csv", rows, output_columns())
    summary_columns = ["family", "target", "condition", "split", "seed_count"]
    summary_columns.extend(
        f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")
    )
    write_csv(
        output_dir / "summary_mean_std.csv", aggregate_metrics(rows), summary_columns
    )


def write_threshold_tables(
    output_dir: Path, thresholds: list[CsvRecord], curves: list[CsvRecord]
) -> None:
    """Write selected validation cutoffs and the full validation curves."""
    write_csv(
        output_dir / "selected_thresholds.csv",
        thresholds,
        [
            "family",
            "target",
            "condition",
            "seed",
            "selected_threshold",
            "validation_CSS",
            "validation_TSS",
            "validation_HSS",
            "validation_POD",
            "validation_prediction_file",
            "test_prediction_file",
            "checkpoint_path",
        ],
    )
    write_csv(
        output_dir / "validation_threshold_curves.csv",
        curves,
        [
            "family",
            "target",
            "condition",
            "seed",
            "threshold",
            *CURVE_METRIC_NAMES,
            "selected",
        ],
    )


def write_physical_threshold_table(output_dir: Path, rows: list[CsvRecord]) -> None:
    """Write the additional max_xflux M1 physical-cutoff evaluation."""
    columns = [
        "family",
        "target",
        "condition",
        "seed",
        "split",
        "threshold",
        "threshold_rule",
        "n_unmatched_split_rows",
        "prediction_file",
        *METRIC_NAMES,
    ]
    write_csv(output_dir / "max_xflux_physical_threshold_metrics.csv", rows, columns)


def evaluation_metadata(
    specs: list[RunSpec], physical_threshold: float
) -> dict[str, CsvValue]:
    """Describe the analysis rules and included evaluation run counts."""
    return {
        "label_definition": "max_goes_class >= M1.0; validated against label_max",
        "score_column": "q50_raw",
        "threshold_selection": "per-run validation CSS maximum; ties: TSS, HSS, POD, higher threshold",
        "css": "0 if TSS or HSS is negative; otherwise sqrt(TSS * HSS)",
        "target_runs": sum(spec.family == "target" for spec in specs),
        "channel_ablation_runs": sum(
            spec.family == "channel_ablation" for spec in specs
        ),
        "physical_max_xflux_threshold": physical_threshold,
        "physical_threshold_units": "W/m^2",
        "leaky_validation_used": False,
        "test_used_for_threshold_selection": False,
    }


def run_evaluation(args: EvaluationArgs) -> None:
    """Load selected artifacts, evaluate them, and write analysis tables."""
    root = args.root.resolve()
    artifact_dir = args.target_artifacts_dir.resolve()
    index = prediction_index(root, artifact_dir)
    specs = target_specs(root, artifact_dir, index)
    specs.extend(channel_specs(root, args.channel_test_summary.resolve(), index))
    labels = {
        split: split_labels(args.split_dir.resolve() / f"{split}.csv")
        for split in ("validation", "test")
    }
    results = [
        evaluate_spec(root, spec, labels, args.max_xflux_threshold) for spec in specs
    ]
    metric_rows = [row for metrics, _, _, _ in results for row in metrics]
    threshold_rows = [threshold for _, threshold, _, _ in results]
    curve_rows = [row for _, _, curves, _ in results for row in curves]
    physical_rows = [row for _, _, _, physical in results for row in physical]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_tables(output_dir, metric_rows)
    write_threshold_tables(output_dir, threshold_rows, curve_rows)
    write_physical_threshold_table(output_dir, physical_rows)
    metadata = evaluation_metadata(specs, args.max_xflux_threshold)
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _ = logger.info(
        "Saved operational metrics: targets={}, channels={}, split rows={}",
        metadata["target_runs"],
        metadata["channel_ablation_runs"],
        len(metric_rows),
    )


def main() -> None:
    """Run the saved-prediction operational skill evaluation."""
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
