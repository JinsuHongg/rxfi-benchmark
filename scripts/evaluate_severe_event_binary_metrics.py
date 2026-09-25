#!/usr/bin/env python3
"""Evaluate severe M+/X+ event metrics from completed saved artifacts only."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from severe_event_metrics import binary_metrics_from_counts, event_binary_labels, ranking_metrics, select_validation_threshold
except ModuleNotFoundError:  # pytest imports this module as scripts.*
    from scripts.severe_event_metrics import binary_metrics_from_counts, event_binary_labels, ranking_metrics, select_validation_threshold


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "severe_event_binary_metrics"
EVENTS = ("M+", "X+")
METRIC_COLUMNS = ("n", "n_positive", "n_negative", "TP", "FP", "TN", "FN", "precision_positive", "recall_positive", "POD", "F1_positive", "F1_macro", "balanced_accuracy", "specificity", "FPR", "FNR", "TSS", "HSS", "AUROC", "AUPRC")
P1 = {target: ROOT / "outputs" / "project3_exports" / target for target in ("max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi")}
P2_QR = {condition: ROOT / "outputs" / "project2" / f"vit_small_224_{condition}_cumulative_peak_flux_qr" / "predictions" for condition in ("hmi_m", "aia131", "aia193", "all13")}
P2_CLASS = {condition: ROOT / "outputs" / "project2" / f"vit_small_224_{condition}_max_flare_class_ordinal6" for condition in ("hmi_m", "aia131", "aia193", "all13")}


def physical_threshold_allowed(target: str) -> bool:
    return target == "max_peak_flux"


def collapse_confusion_matrix(matrix: np.ndarray, row_labels: list[str], column_labels: list[str], event_definition: str) -> dict[str, float | int]:
    expected = {"FQ", "A", "B", "C", "M", "X"}
    if set(row_labels) != expected or set(column_labels) != expected or matrix.shape != (len(row_labels), len(column_labels)):
        raise ValueError("confusion matrix labels must exactly match the verified six-class mapping")
    true_positive = event_binary_labels(row_labels, event_definition)
    pred_positive = event_binary_labels(column_labels, event_definition)
    tp = fp = tn = fn = 0
    for row, truth in enumerate(true_positive):
        for column, prediction in enumerate(pred_positive):
            value = int(matrix[row, column])
            if truth and prediction: tp += value
            elif not truth and prediction: fp += value
            elif not truth and not prediction: tn += value
            else: fn += value
    return binary_metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_qr(path: Path, split: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    required = {"split", "target_name", "max_flare_class", "q50"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: missing required QR columns {sorted(required)}")
    if any(row["split"] != split or row["target_name"] != target for row in rows):
        raise ValueError(f"{path}: split or target does not match its declared source")
    scores = np.asarray([float(row["q50"]) for row in rows], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: q50 contains non-finite values")
    return np.asarray([row["max_flare_class"] for row in rows]), scores


def evaluate_qr_family(family: str, target: str, condition: str, directory: Path, filename_pattern: str, rows: list[dict[str, Any]], curves: list[dict[str, Any]], physical: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    data = {split: load_qr(directory / filename_pattern.format(split=split), split, target) for split in ("validation", "test", "leaky_validation")}
    for event in EVENTS:
        validation_true = event_binary_labels(data["validation"][0], event)
        selected, curve = select_validation_threshold(validation_true, data["validation"][1])
        for point in curve:
            curves.append({"family": family, "target": target, "input_condition": condition, "event_definition": event, **point})
        threshold = float(selected["selected_threshold"])
        for split, (classes, scores) in data.items():
            truth = event_binary_labels(classes, event)
            decision = binary_metrics_from_counts(
                tp=int(np.sum((truth == 1) & (scores >= threshold))), fp=int(np.sum((truth == 0) & (scores >= threshold))),
                tn=int(np.sum((truth == 0) & (scores < threshold))), fn=int(np.sum((truth == 1) & (scores < threshold))),
            )
            rows.append({"family": family, "target": target, "input_condition": condition, "event_definition": event, "threshold_selection_split": "validation", "threshold_rule": "maximize_validation_TSS_then_F1_positive_then_POD_then_higher_threshold", "selected_threshold": threshold, "validation_TSS_at_selection": selected["validation_TSS_at_selection"], "evaluation_split": split, **decision, **ranking_metrics(truth, scores), "score_column": "q50", "source_prediction_file": str((directory / filename_pattern.format(split=split)).relative_to(ROOT))})
        if physical_threshold_allowed(target):
            cutoff = 1e-5 if event == "M+" else 1e-4
            for split in ("validation", "test", "leaky_validation"):
                raw_rows = read_csv(directory / filename_pattern.format(split=split))
                raw_scores = np.asarray([float(row["q50_raw"]) for row in raw_rows], dtype=float)
                truth = event_binary_labels(data[split][0], event)
                decision = binary_metrics_from_counts(tp=int(np.sum((truth == 1) & (raw_scores >= cutoff))), fp=int(np.sum((truth == 0) & (raw_scores >= cutoff))), tn=int(np.sum((truth == 0) & (raw_scores < cutoff))), fn=int(np.sum((truth == 1) & (raw_scores < cutoff))))
                physical.append({"family": family, "target": target, "input_condition": condition, "event_definition": event, "physical_threshold": cutoff, "physical_threshold_description": "M1" if event == "M+" else "X1", "evaluation_split": split, **decision, **ranking_metrics(truth, raw_scores), "score_column": "q50_raw"})
    manifest.append({"family": family, "target": target, "input_condition": condition, "artifact_type": "qr_predictions", "split": "validation,test,leaky_validation", "availability": "available", "reason": "saved q50 prediction CSVs"})


def evaluate_classification(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    for condition, directory in P2_CLASS.items():
        mapping = json.loads((directory / "class_mapping.json").read_text())
        expected_order = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
        matrix_rows = read_csv(directory / "test_confusion_matrix.csv")
        column_labels = list(matrix_rows[0])[1:]
        row_labels = [row["true_class"] for row in matrix_rows]
        if set(expected_order) != set(row_labels) or set(expected_order) != set(column_labels):
            raise ValueError(f"{directory}: confusion matrix labels disagree with class_mapping.json")
        matrix = np.asarray([[int(row[label]) for label in column_labels] for row in matrix_rows], dtype=int)
        for event in EVENTS:
            rows.append({"split": "test", "input_condition": condition, "event_definition": event, **collapse_confusion_matrix(matrix, row_labels, column_labels, event), "AUROC": float("nan"), "AUPRC": float("nan"), "availability_note": "AUROC/AUPRC unavailable: no saved row-level probabilities or logits", "source_confusion_matrix": str((directory / "test_confusion_matrix.csv").relative_to(ROOT))})
        manifest.append({"family": "project2_classification", "target": "max_flare_class_ordinal6", "input_condition": condition, "artifact_type": "corrected_classification_confusion_matrix", "split": "test", "availability": "available", "reason": "verified saved six-class confusion matrix"})
        for split in ("validation", "leaky_validation"):
            manifest.append({"family": "project2_classification", "target": "max_flare_class_ordinal6", "input_condition": condition, "artifact_type": "binary_severe_metrics", "split": split, "availability": "unavailable", "reason": "no saved row-level predictions or confusion matrix"})


def compact(rows: list[dict[str, Any]], destination: Path, family: str) -> None:
    filtered = [row for row in rows if row["family"] == family and row["evaluation_split"] == "test"]
    write_csv(destination, filtered, list(filtered[0]) if filtered else [])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    classification: list[dict[str, Any]] = []; regression: list[dict[str, Any]] = []; physical: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []; manifest: list[dict[str, Any]] = []
    evaluate_classification(classification, manifest)
    for target, directory in P1.items(): evaluate_qr_family("project1_qr", target, "all13", directory, "{split}.csv", regression, curves, physical, manifest)
    for condition, directory in P2_QR.items(): evaluate_qr_family("project2_qr", "cumulative_peak_flux", condition, directory, "{split}_predictions.csv", regression, curves, physical, manifest)
    write_csv(OUTPUT / "classification_binary_severe_event_metrics.csv", classification, ["split", "input_condition", "event_definition", *METRIC_COLUMNS, "availability_note", "source_confusion_matrix"])
    write_csv(OUTPUT / "regression_validation_threshold_metrics.csv", regression, ["family", "target", "input_condition", "event_definition", "threshold_selection_split", "threshold_rule", "selected_threshold", "validation_TSS_at_selection", "evaluation_split", *METRIC_COLUMNS, "score_column", "source_prediction_file"])
    write_csv(OUTPUT / "physical_threshold_evaluation.csv", physical, ["family", "target", "input_condition", "event_definition", "physical_threshold", "physical_threshold_description", "evaluation_split", *METRIC_COLUMNS, "score_column"])
    write_csv(OUTPUT / "validation_threshold_curves.csv", curves, ["family", "target", "input_condition", "event_definition", "threshold", "TP", "FP", "TN", "FN", "POD", "precision_positive", "F1_positive", "F1_macro", "TSS", "HSS", "balanced_accuracy"])
    write_csv(OUTPUT / "artifact_availability.csv", manifest, ["family", "target", "input_condition", "artifact_type", "split", "availability", "reason"])
    compact(regression, OUTPUT / "project1_qr_test_summary.csv", "project1_qr")
    compact(regression, OUTPUT / "project2_qr_test_summary.csv", "project2_qr")
    write_csv(OUTPUT / "project2_classification_test_summary.csv", classification, list(classification[0]))


if __name__ == "__main__":
    main()
