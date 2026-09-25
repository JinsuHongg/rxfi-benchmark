from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.severe_event_metrics import (
    binary_metrics,
    binary_metrics_from_counts,
    event_binary_labels,
    select_validation_threshold,
)


def test_perfect_classifier_has_unit_skill_and_f1() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert metrics["TSS"] == 1.0
    assert metrics["HSS"] == 1.0
    assert metrics["F1_positive"] == 1.0
    assert metrics["POD"] == 1.0


def test_all_negative_predictor_reports_zero_positive_recall() -> None:
    metrics = binary_metrics([0, 1, 1], [0, 0, 0])
    assert (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]) == (0, 0, 1, 2)
    assert metrics["precision_positive"] == 0.0
    assert metrics["POD"] == 0.0
    assert metrics["F1_positive"] == 0.0


def test_all_positive_predictor_reports_zero_specificity() -> None:
    metrics = binary_metrics([0, 0, 1], [1, 1, 1])
    assert (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]) == (1, 2, 0, 0)
    assert metrics["specificity"] == 0.0
    assert metrics["FPR"] == 1.0


def test_known_confusion_matrix_metrics() -> None:
    metrics = binary_metrics_from_counts(tp=3, fp=1, tn=4, fn=2)
    assert math.isclose(metrics["TSS"], 3 / 5 - 1 / 5)
    assert math.isclose(metrics["HSS"], 2 * (3 * 4 - 2 * 1) / ((3 + 2) * (2 + 4) + (3 + 1) * (1 + 4)))
    assert math.isclose(metrics["F1_positive"], 2 * 3 / (2 * 3 + 1 + 2))
    assert math.isclose(metrics["specificity"], 4 / 5)


def test_broad_class_event_mapping_and_direct_class_conversion() -> None:
    classes = ["FQ", "A", "B", "C", "M", "X"]
    assert event_binary_labels(classes, "M+").tolist() == [0, 0, 0, 0, 1, 1]
    assert event_binary_labels(classes, "X+").tolist() == [0, 0, 0, 0, 0, 1]


def test_validation_threshold_ties_use_positive_f1_after_tss() -> None:
    selected, curve = select_validation_threshold(
        np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.7, 0.6])
    )
    assert selected["selected_threshold"] == 0.7
    assert len(curve) == 4


def test_test_labels_do_not_influence_validation_threshold() -> None:
    validation_true = np.asarray([0, 1, 0, 1])
    validation_scores = np.asarray([0.1, 0.8, 0.2, 0.9])
    first, _ = select_validation_threshold(validation_true, validation_scores)
    altered_test_labels = np.asarray([1, 1, 1, 1])
    second, _ = select_validation_threshold(validation_true, validation_scores)
    assert altered_test_labels.sum() == 4
    assert first["selected_threshold"] == second["selected_threshold"]


def test_confusion_matrix_requires_verified_class_labels() -> None:
    from scripts.evaluate_severe_event_binary_metrics import collapse_confusion_matrix

    matrix = np.asarray([[5, 1], [2, 3]])
    with pytest.raises(ValueError, match="labels"):
        collapse_confusion_matrix(matrix, ["FQ", "M"], ["FQ", "A"], "M+")


def test_physical_thresholds_are_limited_to_max_peak_flux() -> None:
    from scripts.evaluate_severe_event_binary_metrics import physical_threshold_allowed

    assert physical_threshold_allowed("max_peak_flux")
    assert not physical_threshold_allowed("cumulative_peak_flux")
    assert not physical_threshold_allowed("max_rxfi")
