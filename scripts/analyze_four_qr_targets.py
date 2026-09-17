#!/usr/bin/env python3
"""Compare completed four-target QR outputs; this script never trains or changes models."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "four_qr_target_analysis"
TARGETS = {
    "max_peak_flux": ROOT / "outputs" / "vit_small_224_max_peak_flux_qr",
    "cumulative_peak_flux": ROOT / "outputs" / "vit_small_224_cumulative_peak_flux_qr",
    "max_rxfi": ROOT / "outputs" / "vit_small_224_max_rxfi_qr",
    "cumulative_rxfi": ROOT / "outputs" / "vit_small_224_cumulative_rxfi_qr",
}
DISPLAY = {
    "max_peak_flux": "QR-1 max peak flux",
    "cumulative_peak_flux": "QR-2 cumulative peak flux",
    "max_rxfi": "QR-3 max RXFI",
    "cumulative_rxfi": "QR-4 cumulative RXFI",
}
KEYS = ["split", "original_row_index", "timestamp"]
CLASS_ORDER = ["FQ", "A", "B", "C", "M", "X"]


def class_band(value: str) -> str:
    """Map detailed GOES labels (for example M2.4) to conventional bands."""
    value = str(value).strip().upper()
    if value == "FQ":
        return "FQ"
    match = re.match(r"^([ABCMX])(?:[0-9].*)?$", value)
    if not match:
        raise ValueError(f"Unsupported max_flare_class value: {value}")
    return match.group(1)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required completed artifact is missing: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def transform(raw: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    name = spec["name"]
    raw = np.asarray(raw, dtype=float)
    if name == "log10_scaled":
        if np.any(raw <= 0):
            raise ValueError("log10_scaled requires strictly positive raw targets")
        return np.log10(raw / float(spec["scale"]))
    if name == "log10_1p_scaled":
        if np.any(raw < 0):
            raise ValueError("log10_1p_scaled requires non-negative raw targets")
        return np.log10(1 + raw / float(spec["scale"]))
    if name == "log10_1p":
        if np.any(raw < 0):
            raise ValueError("log10_1p requires non-negative raw targets")
        return np.log10(1 + raw)
    raise ValueError(f"Unsupported transform: {name}")


def inverse(z: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if spec["name"] == "log10_scaled":
        return float(spec["scale"]) * np.power(10.0, z)
    if spec["name"] == "log10_1p_scaled":
        return float(spec["scale"]) * (np.power(10.0, z) - 1)
    if spec["name"] == "log10_1p":
        return np.power(10.0, z) - 1
    raise ValueError(f"Unsupported transform: {spec['name']}")


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    error = y - pred
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if denominator == 0 else 1 - float(np.sum((y - pred) ** 2)) / denominator


def metric_row(target: str, frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    y, pred = frame.target_transformed.to_numpy(float), frame.q50.to_numpy(float)
    raw, raw_pred = frame.target_raw.to_numpy(float), frame.q50_raw.to_numpy(float)
    iqr = float(np.quantile(y, .75) - np.quantile(y, .25))
    if iqr == 0:
        raise ValueError(f"IQR is zero for {target} on {cohort}; normalized MAE is undefined")
    corr = spearmanr(y, pred)
    tau = kendalltau(y, pred)
    mae = float(np.mean(np.abs(y - pred)))
    return {
        "target": target, "target_label": DISPLAY[target], "cohort": cohort, "n_test": len(frame),
        "spearman": float(corr.statistic), "spearman_pvalue": float(corr.pvalue),
        "kendall_tau": float(tau.statistic), "r2": r2(y, pred),
        "mae_transformed": mae, "rmse_transformed": float(np.sqrt(np.mean((y - pred) ** 2))),
        "iqr_transformed": iqr, "normalized_mae_iqr": mae / iqr,
        "mae_raw": float(np.mean(np.abs(raw - raw_pred))),
        "rmse_raw": float(np.sqrt(np.mean((raw - raw_pred) ** 2))),
    }


def event_rows(target: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rank = {label: index for index, label in enumerate(CLASS_ORDER)}
    unexpected = sorted(set(frame.class_band) - set(rank))
    if unexpected:
        raise ValueError(f"Unknown max_flare_class values: {unexpected}")
    rows = []
    for event, minimum in [("C+", "C"), ("M+", "M"), ("X+", "X")]:
        outcome = (frame.class_band.map(rank) >= rank[minimum]).to_numpy(dtype=int)
        positives, n = int(outcome.sum()), len(outcome)
        prevalence = positives / n
        auroc = float("nan") if positives in (0, n) else float(roc_auc_score(outcome, frame.q50))
        auprc = float("nan") if positives == 0 else float(average_precision_score(outcome, frame.q50))
        rows.append({"target": target, "target_label": DISPLAY[target], "cohort": "common-cohort comparison",
                     "event": event, "n": n, "positives": positives, "prevalence": prevalence,
                     "auroc": auroc, "auprc": auprc, "no_skill_auprc": prevalence,
                     "score": "q50 transformed"})
    return rows


def tail_rows(target: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for tail in (.10, .05):
        cutoff = float(np.quantile(frame.target_transformed, 1 - tail))
        subset = frame.loc[frame.target_transformed >= cutoff]
        y, pred = subset.target_transformed.to_numpy(float), subset.q50.to_numpy(float)
        corr = spearmanr(y, pred) if len(subset) >= 3 else None
        iqr = float(np.quantile(y, .75) - np.quantile(y, .25)) if len(subset) else float("nan")
        mae = float(np.mean(np.abs(y - pred))) if len(subset) else float("nan")
        rows.append({"target": target, "tail": f"top_{int(tail * 100)}pct", "n": len(subset),
                     "true_transformed_cutoff": cutoff,
                     "spearman": float(corr.statistic) if corr else float("nan"),
                     "mae_transformed": mae,
                     "normalized_mae_iqr": mae / iqr if iqr > 0 else float("nan"),
                     "note": "IQR is zero or subset too small" if iqr == 0 or len(subset) < 3 else ""})
    return rows


def make_figures(common: dict[str, pd.DataFrame], learnability: pd.DataFrame,
                 events: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    order = list(TARGETS)
    labels = [DISPLAY[name].replace("QR-", "") for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    data = learnability.set_index("target").loc[order]
    axes[0].bar(labels, data.spearman, color="#4477AA"); axes[0].set_ylabel("Spearman correlation")
    axes[1].bar(labels, data.normalized_mae_iqr, color="#CC6677"); axes[1].set_ylabel("MAE / true-target IQR")
    for ax in axes: ax.tick_params(axis="x", rotation=25); ax.set_title("Common test cohort")
    fig.savefig(OUTPUT / "figure_learnability_comparison.svg"); plt.close(fig)

    for event in ("M+", "X+"):
        subset = events.loc[events.event == event].set_index("target").loc[order]
        if subset.positives.iloc[0] == 0: continue
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.bar(labels, subset.auprc, color="#228833", label="AUPRC")
        ax.axhline(subset.no_skill_auprc.iloc[0], color="black", linestyle="--", label="no-skill prevalence")
        ax.set(title=f"{event} event-ranking relevance (common test cohort)", ylabel="Average precision")
        ax.tick_params(axis="x", rotation=25); ax.legend()
        fig.savefig(OUTPUT / f"figure_event_relevance_{event.replace('+', 'plus')}_auprc.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, target in zip(axes.flat, order):
        frame = common[target]
        classes = [x for x in CLASS_ORDER if x in set(frame.class_band)]
        data = [frame.loc[frame.class_band == label, "q50"].to_numpy() for label in classes]
        ax.boxplot(data, tick_labels=classes, showfliers=False)
        ax.set(title=DISPLAY[target], xlabel="Actual conventional max flare class", ylabel="Predicted q50 (transformed)")
    fig.savefig(OUTPUT / "figure_q50_by_flare_class.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for target in order:
        history = histories[target]
        ax.plot(history.epoch, history.validation_pinball_loss, marker="o", label=DISPLAY[target])
    ax.set(xlabel="Epoch", ylabel="Validation pinball loss", title="Validation learning curves (target scales differ)")
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT / "figure_validation_learning_curves.svg"); plt.close(fig)


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tests: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    configs: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    summaries: dict[str, pd.DataFrame] = {}
    validation: dict[str, pd.DataFrame] = {}
    leaky_validation: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, str]] = []
    for target, directory in TARGETS.items():
        configs[target] = yaml.safe_load(require(directory / "resolved_config.yaml").read_text())
        checkpoints[target] = json.loads(require(directory / "checkpoint_metadata.json").read_text())
        histories[target] = read_csv(directory / "training_log.csv")
        summaries[target] = read_csv(directory / "summary_metrics.csv")
        selected_epochs = set(summaries[target]["selected_epoch"].astype(int))
        if selected_epochs != {int(checkpoints[target]["best_validation_epoch"])}:
            raise ValueError(f"Summary selected epoch does not match best checkpoint for {target}")
        for split, destination in [("test", tests), ("validation", validation), ("leaky_validation", leaky_validation)]:
            path = directory / "predictions" / f"{split}_predictions.csv"
            frame = read_csv(path)
            if frame.split.nunique() != 1 or frame.split.iloc[0] != split:
                raise ValueError(f"Unexpected split in {path}")
            if frame.duplicated(KEYS).any():
                raise ValueError(f"Duplicate stable keys in {path}")
            destination[target] = frame
            manifest.append({"target": target, "split": split, "prediction_file": str(path.relative_to(ROOT)),
                             "selected_best_epoch": str(checkpoints[target]["best_validation_epoch"])})
        spec = configs[target]["target_transform"]
        recomputed = transform(tests[target].target_raw.to_numpy(float), spec)
        if not np.allclose(recomputed, tests[target].target_transformed, rtol=1e-9, atol=1e-10):
            raise ValueError(f"Stored target transform mismatch for {target}")
        raw_from_q50 = inverse(tests[target].q50.to_numpy(float), spec)
        if not np.allclose(raw_from_q50, tests[target].q50_raw.to_numpy(float), rtol=1e-9, atol=1e-10):
            raise ValueError(f"Stored q50 inverse-transform mismatch for {target}")

    key_sets = {target: set(map(tuple, frame[KEYS].itertuples(index=False, name=None))) for target, frame in tests.items()}
    common_keys = set.intersection(*key_sets.values())
    if not common_keys:
        raise ValueError("Common test cohort is empty")
    common: dict[str, pd.DataFrame] = {}
    cohort_rows = []
    for target, frame in tests.items():
        mask = frame[KEYS].apply(tuple, axis=1).isin(common_keys)
        common[target] = frame.loc[mask].sort_values(KEYS).reset_index(drop=True)
        cohort_rows.append({"target": target, "full_test_n": len(frame), "common_test_n": len(common[target]),
                            "unique_to_target_n": len(key_sets[target] - set.union(*(sets for name, sets in key_sets.items() if name != target))),
                            "excluded_from_common_n": len(frame) - len(common[target])})
    reference = common[next(iter(TARGETS))][KEYS]
    for target, frame in common.items():
        if not reference.equals(frame[KEYS]):
            raise ValueError(f"Common cohort order/key mismatch for {target}")
        if not common[next(iter(TARGETS))].max_flare_class.equals(frame.max_flare_class):
            raise ValueError(f"Conventional class labels mismatch on common cohort for {target}")
    for frame in common.values():
        frame["class_band"] = frame.max_flare_class.map(class_band)
    class_values = sorted(set(common[next(iter(TARGETS))].class_band), key=CLASS_ORDER.index)

    common_metrics = pd.DataFrame([metric_row(target, common[target], "common-cohort comparison") for target in TARGETS])
    full_metrics = pd.DataFrame([metric_row(target, tests[target], "target-specific full-cohort comparison") for target in TARGETS])
    event_table = pd.DataFrame([row for target in TARGETS for row in event_rows(target, common[target])])
    tail_table = pd.DataFrame([row for target in TARGETS for row in tail_rows(target, common[target])])

    diagnostics, convergence = [], []
    for target in TARGETS:
        summary = summaries[target].set_index("split")
        test = summary.loc["test"] if "test" in summary.index else pd.Series(dtype=float)
        history = histories[target]
        best_epoch = int(checkpoints[target]["best_validation_epoch"])
        best_loss = float(checkpoints[target]["best_validation_pinball_loss"])
        final = float(history.validation_pinball_loss.iloc[-1])
        diagnostics.append({"target": target, "best_epoch": best_epoch, "best_val_pinball": best_loss,
                            "final_val_pinball": final, "test_pinball": test.get("pinball_loss", np.nan),
                            "q05_pinball": test.get("pinball_q05", np.nan), "q50_pinball": test.get("pinball_q50", np.nan),
                            "q95_pinball": test.get("pinball_q95", np.nan), "empirical_q05": test.get("fraction_target_le_q05", np.nan),
                            "empirical_q50": test.get("fraction_target_le_q50", np.nan), "empirical_q95": test.get("fraction_target_le_q95", np.nan),
                            "crossing_q05_q50": test.get("cross_q05_gt_q50", np.nan), "crossing_q50_q95": test.get("cross_q50_gt_q95", np.nan),
                            "crossing_q05_q95": test.get("cross_q05_gt_q95", np.nan)})
        recent = history.validation_pinball_loss.tail(3).tolist()
        convergence.append({"target": target, "total_epochs_completed": len(history), "best_epoch": best_epoch,
                            "best_val_pinball": best_loss, "final_val_pinball": final,
                            "best_epoch_is_final": best_epoch == len(history), "last_three_val_pinball": json.dumps(recent),
                            "logged_training_seconds": float(history.elapsed_seconds.sum()),
                            "possible_undertraining": "review last-three losses" if best_epoch == len(history) else "no: best checkpoint precedes final epoch"})

    pd.DataFrame(manifest).to_csv(OUTPUT / "prediction_artifact_manifest.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(OUTPUT / "table_cohort_accounting.csv", index=False)
    common_metrics.rename(columns={"n_test": "n_test_common"}).to_csv(OUTPUT / "table_learnability_common_cohort.csv", index=False)
    full_metrics.to_csv(OUTPUT / "table_learnability_full_cohort.csv", index=False)
    event_table.to_csv(OUTPUT / "table_event_relevance.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "table_qr_diagnostics.csv", index=False)
    tail_table.to_csv(OUTPUT / "table_tail_learnability.csv", index=False)
    pd.DataFrame(convergence).to_csv(OUTPUT / "table_convergence_runtime.csv", index=False)
    pd.DataFrame({"class": [x for x in CLASS_ORDER if x in class_values]}).to_csv(OUTPUT / "class_order_used.csv", index=False)
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps({
        "analysis": "completed best-checkpoint QR prediction exports only; no retraining or calibration",
        "common_test_cohort_n": len(common_keys), "key_columns": KEYS, "class_order": [x for x in CLASS_ORDER if x in class_values],
        "event_score": "q50 transformed", "event_cohort": "common-cohort comparison",
        "validation": "duplicate keys checked; transforms and q50 raw inverse mappings checked",
    }, indent=2) + "\n")
    make_figures(common, common_metrics, event_table, histories)
    print(json.dumps({"output_dir": str(OUTPUT), "common_test_cohort_n": len(common_keys), "class_values": class_values}, indent=2))


if __name__ == "__main__":
    main()
