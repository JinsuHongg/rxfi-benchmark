"""Shared binary severe-event metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


EVENT_POSITIVE_CLASSES = {"M+": frozenset(("M", "X")), "X+": frozenset(("X",))}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")

def _classification_ratio(numerator: float, denominator: float) -> float:
    """Match sklearn zero_division=0 for precision and F1."""
    return float(numerator / denominator) if denominator else 0.0


def event_binary_labels(classes: Iterable[str], event_definition: str) -> np.ndarray:
    """Collapse saved NOAA broad labels (or magnitude strings) to M+/X+ labels."""
    if event_definition not in EVENT_POSITIVE_CLASSES:
        raise ValueError(f"Unsupported event definition: {event_definition!r}")
    broad: list[str] = []
    for value in classes:
        if value == "FQ":
            broad.append("FQ")
        elif isinstance(value, str) and value and value[0] in {"A", "B", "C", "M", "X"}:
            broad.append(value[0])
        else:
            raise ValueError(f"Invalid broad flare class: {value!r}")
    return np.asarray([int(label in EVENT_POSITIVE_CLASSES[event_definition]) for label in broad], dtype=int)


def binary_metrics_from_counts(*, tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    """Return guarded binary decision metrics from an explicit confusion matrix."""
    precision = _classification_ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1_positive = _classification_ratio(2 * tp, 2 * tp + fp + fn)
    f1_negative = _classification_ratio(2 * tn, 2 * tn + fp + fn)
    macro_f1 = float(np.nanmean([f1_negative, f1_positive]))
    balanced_accuracy = float(np.nanmean([recall, specificity]))
    tss = recall - _ratio(fp, fp + tn)
    hss = _ratio(2 * (tp * tn - fn * fp), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return {
        "n": tp + fp + tn + fn,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision_positive": precision,
        "recall_positive": recall,
        "POD": recall,
        "F1_positive": f1_positive,
        "F1_negative": f1_negative,
        "F1_macro": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "FPR": _ratio(fp, fp + tn),
        "FNR": _ratio(fn, tp + fn),
        "TSS": tss,
        "HSS": hss,
    }


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float | int]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError("y_true and y_pred must be equal-length one-dimensional arrays")
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("binary inputs must contain only 0 and 1")
    return binary_metrics_from_counts(
        tp=int(np.sum((true == 1) & (pred == 1))),
        fp=int(np.sum((true == 0) & (pred == 1))),
        tn=int(np.sum((true == 0) & (pred == 0))),
        fn=int(np.sum((true == 1) & (pred == 0))),
    )


def ranking_metrics(y_true: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    true, score = np.asarray(list(y_true), dtype=int), np.asarray(list(scores), dtype=float)
    if true.shape != score.shape or true.ndim != 1:
        raise ValueError("y_true and scores must be equal-length one-dimensional arrays")
    if not np.isfinite(score).all() or not np.isin(true, [0, 1]).all():
        raise ValueError("scores must be finite and labels must be binary")
    if np.unique(true).size < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan")}
    return {"AUROC": float(roc_auc_score(true, score)), "AUPRC": float(average_precision_score(true, score))}


def threshold_curve(y_true: Iterable[int], scores: Iterable[float]) -> list[dict[str, float | int]]:
    true, score = np.asarray(list(y_true), dtype=int), np.asarray(list(scores), dtype=float)
    if true.shape != score.shape or true.ndim != 1 or not np.isfinite(score).all() or not np.isin(true, [0, 1]).all():
        raise ValueError("threshold inputs require equal-length finite scores")
    order = np.argsort(-score, kind="stable")
    sorted_scores, sorted_true = score[order], true[order]
    positives, negatives = int(true.sum()), int(len(true) - true.sum())
    tp = fp = 0
    curve: list[dict[str, float | int]] = []
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_true[start:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        curve.append({"threshold": float(sorted_scores[start]), **binary_metrics_from_counts(tp=tp, fp=fp, tn=negatives - fp, fn=positives - tp)})
        start = end
    return curve


def select_validation_threshold(y_true: Iterable[int], scores: Iterable[float]) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Maximize validation TSS, then F1+, POD, then choose the higher threshold."""
    curve = threshold_curve(y_true, scores)
    if not curve:
        raise ValueError("cannot select a threshold from no validation predictions")

    def value(row: dict[str, float | int], key: str) -> float:
        candidate = float(row[key])
        return candidate if np.isfinite(candidate) else float("-inf")

    selected = max(curve, key=lambda row: (value(row, "TSS"), value(row, "F1_positive"), value(row, "POD"), float(row["threshold"])))
    return {"selected_threshold": selected["threshold"], "validation_TSS_at_selection": selected["TSS"], **selected}, curve
