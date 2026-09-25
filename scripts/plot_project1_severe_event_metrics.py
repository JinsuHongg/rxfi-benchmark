#!/usr/bin/env python3
"""Create proposal-ready Project 1 severe-event metric comparison figures.

This is a read-only visualization of saved severe-event evaluation results.  It
does not load checkpoints, regenerate predictions, or train models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "severe_event_binary_metrics" / "project1_qr_test_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "severe_event_binary_metrics" / "figures"

TARGET_ORDER = ["max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi"]
TARGET_LABELS = {
    "max_peak_flux": "Max Peak",
    "cumulative_peak_flux": "Cum. Peak",
    "max_rxfi": "Max RXFI",
    "cumulative_rxfi": "Cum. RXFI",
}
METRICS = [("AUPRC", "AUPRC"), ("TSS", "TSS"), ("HSS", "HSS"), ("F1_positive", "Positive F1")]
THRESHOLD_RULE = "maximize_validation_TSS_then_F1_positive_then_POD_then_higher_threshold"


def load_event_metrics(source: Path, event: str) -> tuple[pd.DataFrame, float]:
    """Load and validate the saved TEST metrics for a single severe-event class."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing machine-readable severe-event summary: {source}")
    metrics = pd.read_csv(source)
    required = {
        "family", "target", "event_definition", "evaluation_split", "threshold_selection_split",
        "threshold_rule", "selected_threshold", "validation_TSS_at_selection", "n", "n_positive",
        "n_negative", "AUPRC", "TSS", "HSS", "F1_positive",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns in {source}: {sorted(missing)}")

    subset = metrics.loc[(metrics["family"] == "project1_qr") & (metrics["event_definition"] == event)].copy()
    subset = subset.set_index("target").reindex(TARGET_ORDER).reset_index()
    if subset["target"].isna().any() or subset["evaluation_split"].isna().any():
        raise ValueError(f"Saved Project 1 {event} summary lacks one or more target representations")
    if len(subset) != len(TARGET_ORDER):
        raise ValueError(f"Expected one saved Project 1 {event} row per target; found {len(subset)}")
    if subset["evaluation_split"].ne("test").any():
        raise ValueError("Only TEST metrics may be plotted")
    if subset["threshold_selection_split"].ne("validation").any():
        raise ValueError("Thresholds must be selected on validation only")
    if subset["threshold_rule"].ne(THRESHOLD_RULE).any():
        raise ValueError("Unexpected threshold-selection rule in saved metrics")
    provenance = subset[["selected_threshold", "validation_TSS_at_selection"]].to_numpy(dtype=float)
    if not np.isfinite(provenance).all():
        raise ValueError("Saved selected_threshold and validation_TSS_at_selection must be finite")
    if subset[["n", "n_positive", "n_negative"]].nunique().max() != 1:
        raise ValueError("All target representations must use the same TEST cohort")
    n = int(subset["n"].iloc[0])
    positives = int(subset["n_positive"].iloc[0])
    negatives = int(subset["n_negative"].iloc[0])
    if positives + negatives != n:
        raise ValueError("Saved TEST support counts are inconsistent")
    return subset, positives / n


def _annotate_bars(ax: plt.Axes, bars, values: np.ndarray) -> None:
    best = int(np.argmax(values))
    for index, (bar, value) in enumerate(zip(bars, values)):
        label = f"{value:.3f}" + (" ★" if index == best else "")
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold" if index == best else "normal",
        )


def make_figure(metrics: pd.DataFrame, prevalence: float, event: str) -> plt.Figure:
    """Render a four-panel presentation figure from validated saved metrics."""
    labels = [TARGET_LABELS[target] for target in TARGET_ORDER]
    x = np.arange(len(labels))
    event_prefix = "M+" if event == "M+" else "X+"
    title = (
        "Project 1: Severe-event performance across target representations"
        if event == "M+"
        else "Project 1: X+ severe-event performance"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.80, hspace=0.48, wspace=0.20)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        f"{event_prefix} TEST performance; TSS/HSS/F1 use validation-selected thresholds frozen before TEST evaluation",
        ha="center",
        fontsize=12.5,
    )

    for ax, (column, panel_title) in zip(axes.flat, METRICS):
        values = metrics[column].to_numpy(dtype=float)
        bars = ax.bar(x, values)  # Matplotlib's default color; emphasis comes from annotations only.
        _annotate_bars(ax, bars, values)
        ax.set_title(panel_title, fontsize=17, pad=12)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.88", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if column == "AUPRC":
            ax.set_ylim(0, 0.8)
            ax.axhline(prevalence, color="0.15", linestyle="--", linewidth=1.25, label="No-skill AUPRC")
            ax.legend(loc="lower right", fontsize=11, frameon=False)
        else:
            ax.set_ylim(0, 1)
        for tick, target in zip(ax.get_xticklabels(), TARGET_ORDER):
            if target == "cumulative_peak_flux":
                tick.set_fontweight("bold")

    takeaway = (
        "Cumulative peak flux performs best across ranking and threshold-based M+ metrics."
        if event == "M+"
        else "X+ is more difficult: ranking is above no-skill, while HSS and positive F1 remain low."
    )
    fig.text(0.5, 0.09, takeaway, ha="center", fontsize=14, fontweight="bold" if event == "M+" else "normal")
    fig.text(0.5, 0.045, "★ Best value within panel. Source: saved Project 1 QR severe-event TEST evaluation.", ha="center", fontsize=11)
    return fig


def save_figure(figure: plt.Figure, destination: Path) -> list[Path]:
    """Save slide-ready PNG, PDF, and SVG variants without touching other figures."""
    outputs = []
    for suffix in (".png", ".pdf", ".svg"):
        path = destination.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Saved Project 1 severe-event TEST CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for proposal figure exports")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for event, stem in (("M+", "project1_mplus_metric_comparison"), ("X+", "project1_xplus_metric_comparison")):
        metrics, prevalence = load_event_metrics(args.input, event)
        print(f"\nProject 1 {event} exact TEST values used:")
        print(metrics[["target", "AUPRC", "TSS", "HSS", "F1_positive"]].to_string(index=False, float_format=lambda value: f"{value:.12f}"))
        print(f"{event} no-skill AUPRC (TEST prevalence): {prevalence:.12f}")
        print("Threshold confirmation: TEST metrics; thresholds selected on validation only and frozen before TEST evaluation.")
        created.extend(save_figure(make_figure(metrics, prevalence, event), args.output_dir / stem))

    print("\nFiles created:")
    print("\n".join(str(path) for path in created))


if __name__ == "__main__":
    main()
