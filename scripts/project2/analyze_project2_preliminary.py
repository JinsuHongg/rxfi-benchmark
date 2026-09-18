#!/usr/bin/env python3
"""Build the final Project 2 comparison from completed ordinal6 and QR artifacts.

The original 310-class classifier outputs are intentionally excluded.  Classification
metrics are reported both as saved (six encoded labels) and over observed test classes
only, because the fixed A band has zero support in the current cohort.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "project2"
CHANNELS = ("hmi_m", "aia131", "aia193", "all13")
CLASS_ORDER = ("FQ", "A", "B", "C", "M", "X")
OBSERVED = ("FQ", "B", "C", "M", "X")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float) -> float:
    return float(value)


def band(value: str) -> str:
    if value == "FQ":
        return "FQ"
    if len(value) >= 2 and value[0] in "ABCMX" and value[1:].replace(".", "", 1).isdigit():
        return value[0]
    raise ValueError(f"Unexpected NOAA flare class in prediction export: {value!r}")


def run_dir(channel: str, task: str) -> Path:
    suffix = "max_flare_class_ordinal6" if task == "classification" else "cumulative_peak_flux_qr"
    return OUT / f"vit_small_224_{channel}_{suffix}"


def metric_row(rows: list[dict[str, str]], split: str) -> dict[str, str]:
    selected = [row for row in rows if row["split"] == split]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {split} row, found {len(selected)}")
    return selected[0]


def classification() -> tuple[list[dict], dict[str, np.ndarray], list[dict], list[dict]]:
    summary_rows, matrices, per_class_rows, runtime_rows = [], {}, [], []
    for channel in CHANNELS:
        directory = run_dir(channel, "classification")
        mapping = json.loads((directory / "class_mapping.json").read_text())
        if mapping != {name: index for index, name in enumerate(CLASS_ORDER)}:
            raise ValueError(f"{channel}: corrected ordinal6 mapping not found: {mapping}")
        per_class = read_csv(directory / "test_per_class_metrics.csv")
        by_class = {row["class"]: row for row in per_class}
        if tuple(by_class) != CLASS_ORDER:
            raise ValueError(f"{channel}: unexpected class order {tuple(by_class)}")
        supports = {name: int(by_class[name]["support"]) for name in CLASS_ORDER}
        if supports["A"] != 0 or any(supports[name] == 0 for name in OBSERVED):
            raise ValueError(f"{channel}: expected only A to have zero test support: {supports}")
        observed_f1 = float(np.mean([as_float(by_class[name]["f1"]) for name in OBSERVED]))
        observed_balanced = float(np.mean([as_float(by_class[name]["recall"]) for name in OBSERVED]))
        saved_test = metric_row(read_csv(directory / "summary_metrics.csv"), "test")
        checkpoint = json.loads((directory / "checkpoint_metadata.json").read_text())
        history = read_csv(directory / "training_log.csv")
        final = history[-1]
        summary_rows.append({
            "input": channel, "n_test": int(saved_test["n_samples"]),
            "accuracy": as_float(saved_test["accuracy"]),
            "balanced_accuracy_observed": observed_balanced,
            "macro_f1_observed": observed_f1,
            "balanced_accuracy_saved": as_float(saved_test["balanced_accuracy"]),
            "macro_f1_saved_six_class": as_float(saved_test["macro_f1"]),
            "macro_average_classes": "+".join(OBSERVED),
            **{f"{name.lower()}_recall": as_float(by_class[name]["recall"]) for name in CLASS_ORDER},
            "best_epoch": int(checkpoint["best_validation_epoch"]),
            "best_val_macro_f1_saved_six_class": as_float(checkpoint["best_validation_macro_f1"]),
            "final_epoch": int(final["epoch"]),
            "final_val_macro_f1_saved_six_class": as_float(final["validation_macro_f1"]),
        })
        for row in per_class:
            per_class_rows.append({"input": channel, **row})
        matrix_rows = read_csv(directory / "test_confusion_matrix.csv")
        matrices[channel] = np.array([[int(row[name]) for name in CLASS_ORDER] for row in matrix_rows])
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        smoke = json.loads((directory / "smoke_test.json").read_text())
        runtime_rows.append({
            "task": "classification_ordinal6", "input": channel,
            "runtime_seconds_epoch_sum": sum(as_float(row["elapsed_seconds"]) for row in history),
            "runtime_hours_epoch_sum": sum(as_float(row["elapsed_seconds"]) for row in history) / 3600,
            "runtime_source": "training_log elapsed_seconds sum (excludes setup/evaluation)",
            "best_epoch": int(checkpoint["best_validation_epoch"]),
            "gpu": smoke.get("device", "NA"), "batch_size": config["batch_size"],
            "precision": config["precision"], "num_workers": config["num_workers"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        })
    return summary_rows, matrices, per_class_rows, runtime_rows


def regression() -> tuple[list[dict], list[dict]]:
    result, runtime_rows = [], []
    for channel in CHANNELS:
        directory = run_dir(channel, "regression")
        prediction_path = directory / "predictions" / "test_predictions.csv"
        predictions = read_csv(prediction_path)
        keys = [(r["split"], r["original_row_index"], r["timestamp"]) for r in predictions]
        if len(keys) != len(set(keys)) or len(predictions) != 27980:
            raise ValueError(f"{channel}: expected 27,980 unique test prediction keys")
        y = np.array([as_float(r["target_transformed"]) for r in predictions])
        score = np.array([as_float(r["q50"]) for r in predictions])
        iqr = float(np.quantile(y, .75) - np.quantile(y, .25))
        if iqr <= 0:
            raise ValueError(f"{channel}: non-positive target IQR")
        bands = np.array([band(r["max_flare_class"]) for r in predictions])
        mplus = np.isin(bands, ["M", "X"]).astype(int)
        xplus = (bands == "X").astype(int)
        if len(np.unique(mplus)) != 2 or len(np.unique(xplus)) != 2:
            raise ValueError(f"{channel}: severe-event outcome lacks both classes")
        result.append({
            "input": channel, "n_test": len(predictions),
            "spearman": float(spearmanr(y, score).statistic),
            "r2": float(r2_score(y, score)),
            "mae_iqr": float(np.mean(np.abs(y - score)) / iqr),
            "mplus_auprc": float(average_precision_score(mplus, score)),
            "mplus_auroc": float(roc_auc_score(mplus, score)),
            "xplus_auprc": float(average_precision_score(xplus, score)),
            "xplus_auroc": float(roc_auc_score(xplus, score)),
            "mplus_prevalence": float(mplus.mean()), "xplus_prevalence": float(xplus.mean()),
        })
        summary = metric_row(read_csv(directory / "summary_metrics.csv"), "test")
        checkpoint = json.loads((directory / "checkpoint_metadata.json").read_text())
        history = read_csv(directory / "training_log.csv")
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        meta = json.loads((directory / "experiment_metadata.json").read_text())
        runtime_rows.append({
            "task": "cumulative_peak_flux_qr", "input": channel,
            "runtime_seconds_epoch_sum": sum(as_float(row["elapsed_seconds"]) for row in history),
            "runtime_hours_epoch_sum": sum(as_float(row["elapsed_seconds"]) for row in history) / 3600,
            "runtime_source": "training_log elapsed_seconds sum (excludes setup/evaluation)",
            "best_epoch": int(checkpoint["best_validation_epoch"]),
            "gpu": meta.get("gpu_name", "NA"), "batch_size": config["batch_size"],
            "precision": config["precision"], "num_workers": config["num_workers"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "test_pinball_loss": as_float(summary["pinball_loss"]),
        })
    return result, runtime_rows


def grouped_bar(path: Path, labels: list[str], values: list[list[float]], names: list[str], ylabel: str, baselines: list[float] | None = None) -> None:
    x = np.arange(len(labels)); width = .34 if len(values) == 2 else .22
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (series, name) in enumerate(zip(values, names)):
        ax.bar(x + (i - (len(values)-1)/2) * width, series, width, label=name)
    if baselines:
        for baseline, name in zip(baselines, names): ax.axhline(baseline, ls="--", lw=1, color="0.35", label=f"{name} no-skill")
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0); ax.legend(frameon=False, ncols=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_confusion(path: Path, matrices: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, channel in zip(axes.flat, CHANNELS):
        matrix = matrices[channel]
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(6):
            for j in range(6): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
        ax.set(title=channel, xlabel="Predicted class", ylabel="True class", xticks=range(6), yticks=range(6), xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label="Test samples")
    fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cls, matrices, per_class, cls_runtime = classification()
    qr, qr_runtime = regression()
    write_csv(OUT / "table_classification_ordinal6_test.csv", cls)
    write_csv(OUT / "table_classification_ordinal6_per_class_test.csv", per_class)
    write_csv(OUT / "table_qr_cumulative_peak_flux_test.csv", qr)
    integrated = []
    for c, q in zip(cls, qr):
        integrated.append({"input": c["input"], "n_test": c["n_test"], "macro_f1": c["macro_f1_observed"], "balanced_accuracy": c["balanced_accuracy_observed"], "m_recall": c["m_recall"], "x_recall": c["x_recall"], **{k: q[k] for k in ("spearman", "r2", "mae_iqr", "mplus_auprc", "mplus_auroc", "xplus_auprc", "xplus_auroc")}})
    write_csv(OUT / "table_project2_preliminary_summary.csv", integrated)
    write_csv(OUT / "table_runtime_summary.csv", cls_runtime + qr_runtime)
    write_csv(OUT / "table_classification_all13_deltas.csv", [{"baseline": base["input"], **{f"delta_{metric}": cls[-1][metric] - base[metric] for metric in ("macro_f1_observed", "balanced_accuracy_observed", "m_recall", "x_recall")}} for base in cls[:-1]])
    strongest = max(qr[:-1], key=lambda row: row["mplus_auprc"])
    write_csv(OUT / "table_qr_all13_delta_vs_strongest_single_channel.csv", [{"strongest_single_channel": strongest["input"], **{f"delta_{metric}": qr[-1][metric] - strongest[metric] for metric in ("spearman", "r2", "mae_iqr", "mplus_auprc", "mplus_auroc", "xplus_auprc", "xplus_auroc")}}])
    labels = list(CHANNELS)
    grouped_bar(OUT / "figure_classification_ordinal6_comparison.png", labels, [[r["macro_f1_observed"] for r in cls], [r["balanced_accuracy_observed"] for r in cls]], ["Macro-F1 (observed)", "Balanced accuracy (observed)"], "Score")
    grouped_bar(OUT / "figure_classification_ordinal6_rare_recall.png", labels, [[r["m_recall"] for r in cls], [r["x_recall"] for r in cls]], ["M recall", "X recall"], "Recall")
    grouped_bar(OUT / "figure_qr_learnability_comparison.png", labels, [[r["spearman"] for r in qr], [r["mae_iqr"] for r in qr]], ["Spearman", "MAE / IQR"], "Metric value")
    grouped_bar(OUT / "figure_qr_event_ranking_comparison.png", labels, [[r["mplus_auprc"] for r in qr], [r["xplus_auprc"] for r in qr]], ["M+ AUPRC", "X+ AUPRC"], "AUPRC", [qr[0]["mplus_prevalence"], qr[0]["xplus_prevalence"]])
    plot_confusion(OUT / "figure_classification_ordinal6_confusion_matrices.png", matrices)
    print(json.dumps({"status": "complete", "n_test": 27980, "classification_inputs": list(CHANNELS), "qr_inputs": list(CHANNELS), "old_310_class_outputs_used": False}, indent=2))


if __name__ == "__main__":
    main()
