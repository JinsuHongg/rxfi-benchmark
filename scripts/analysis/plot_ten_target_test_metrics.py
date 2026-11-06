# pyright: reportUnknownMemberType=false
"""Plot final test metrics for the ten V1 targets."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

TARGET_LABELS = {
    "fluence": "Fluence",
    "max_xflux": "Maximum X-ray flux",
    "max_background_deviation_2h_q10": "Max background deviation · 2 h · q10",
    "max_background_deviation_2h_q50": "Max background deviation · 2 h · q50",
    "max_background_deviation_24h_q10": "Max background deviation · 24 h · q10",
    "max_background_deviation_24h_q50": "Max background deviation · 24 h · q50",
    "total_background_deviation_2h_q10": "Total background deviation · 2 h · q10",
    "total_background_deviation_2h_q50": "Total background deviation · 2 h · q50",
    "total_background_deviation_24h_q10": "Total background deviation · 24 h · q10",
    "total_background_deviation_24h_q50": "Total background deviation · 24 h · q50",
}
METRIC_COLUMNS = {
    "pinball_q50": "pinball_q50",
    "spearman_q50": "spearman_q50",
    "r2_q50_transformed": "r2_q50_transformed",
}
SEED_MARKERS = {1: "o", 2: "s", 3: "^"}
SEED_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}


@dataclass(frozen=True)
class TestRun:
    """Hold the plotted metrics for one target and random seed."""

    target: str
    seed: int
    sample_count: int
    pinball_q50: float
    spearman_q50: float
    r2_q50_transformed: float


@dataclass(frozen=True)
class TargetSummary:
    """Summarize seed-level test metrics for one target."""

    target: str
    sample_count: int
    runs: tuple[TestRun, ...]
    pinball_mean: float
    pinball_sd: float
    spearman_mean: float
    spearman_sd: float
    r2_mean: float
    r2_sd: float


@dataclass(frozen=True)
class PlotArguments:
    """Store typed command-line arguments for plotting."""

    input_csv: Path
    output_dir: Path
    expected_seeds: int


def repository_root() -> Path:
    """Return the repository root from this script's location."""
    return Path(__file__).resolve().parents[2]


def parse_number(row: dict[str, str], column: str, line_number: int) -> float:
    """Read a finite numeric CSV value with a useful error message."""
    raw_value = row.get(column, "")
    if raw_value == "":
        raise ValueError(f"Missing {column!r} on CSV line {line_number}")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column!r} on CSV line {line_number}")
    return value


def load_test_runs(input_csv: Path, expected_seeds: int) -> list[TestRun]:
    """Load and validate test results selected using validation data."""
    runs: list[TestRun] = []
    with input_csv.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for line_number, row in enumerate(reader, start=2):
            if row["model"] != "compact_cnn_v1":
                raise ValueError(f"Unexpected model {row['model']!r} on line {line_number}")
            if row["checkpoint_selection_split"] != "validation":
                raise ValueError(
                    f"Checkpoint on line {line_number} was not selected on validation"
                )
            target = row["target"]
            if target not in TARGET_LABELS:
                raise ValueError(f"Unexpected target {target!r} on line {line_number}")
            runs.append(
                TestRun(
                    target=target,
                    seed=int(row["seed"]),
                    sample_count=int(row["test_sample_count"]),
                    pinball_q50=parse_number(row, METRIC_COLUMNS["pinball_q50"], line_number),
                    spearman_q50=parse_number(row, METRIC_COLUMNS["spearman_q50"], line_number),
                    r2_q50_transformed=parse_number(
                        row, METRIC_COLUMNS["r2_q50_transformed"], line_number
                    ),
                )
            )
    validate_runs(runs, expected_seeds)
    return runs


def validate_runs(runs: list[TestRun], expected_seeds: int) -> None:
    """Require all ten targets to have unique, complete seed results."""
    grouped: dict[str, list[TestRun]] = {target: [] for target in TARGET_LABELS}
    for run in runs:
        grouped[run.target].append(run)
    if {run.target for run in runs} != set(TARGET_LABELS):
        raise ValueError("The test CSV does not contain all ten expected targets")
    for target, target_runs in grouped.items():
        seeds = [run.seed for run in target_runs]
        if len(seeds) != expected_seeds or len(set(seeds)) != expected_seeds:
            raise ValueError(
                f"{target} has seeds {seeds}; expected {expected_seeds} unique seeds"
            )
        sample_counts = {run.sample_count for run in target_runs}
        if len(sample_counts) != 1:
            raise ValueError(f"{target} has inconsistent test sample counts")


def summarize_runs(runs: list[TestRun]) -> list[TargetSummary]:
    """Compute means and sample standard deviations across seeds."""
    grouped: dict[str, list[TestRun]] = {target: [] for target in TARGET_LABELS}
    for run in runs:
        grouped[run.target].append(run)
    summaries = [
        TargetSummary(
            target=target,
            sample_count=target_runs[0].sample_count,
            runs=tuple(sorted(target_runs, key=lambda item: item.seed)),
            pinball_mean=mean(run.pinball_q50 for run in target_runs),
            pinball_sd=stdev(run.pinball_q50 for run in target_runs),
            spearman_mean=mean(run.spearman_q50 for run in target_runs),
            spearman_sd=stdev(run.spearman_q50 for run in target_runs),
            r2_mean=mean(run.r2_q50_transformed for run in target_runs),
            r2_sd=stdev(run.r2_q50_transformed for run in target_runs),
        )
        for target, target_runs in grouped.items()
    ]
    return sorted(summaries, key=lambda item: item.pinball_mean)


def configure_plot_style() -> None:
    """Set readable, editable vector-font defaults."""
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_seed_points(
    axis: Axes,
    summaries: list[TargetSummary],
    metric: str,
    metric_sd: str,
) -> None:
    """Draw seed observations and mean ± sample SD for one metric."""
    offsets = {1: -0.17, 2: 0.0, 3: 0.17}
    for position, summary in enumerate(summaries):
        for run in summary.runs:
            _ = axis.scatter(
                run_metric(run, metric),
                position + offsets[run.seed],
                color=SEED_COLORS[run.seed],
                marker=SEED_MARKERS[run.seed],
                s=28,
                zorder=3,
            )
        _ = axis.errorbar(
            summary_metric(summary, metric),
            position,
            xerr=summary_uncertainty(summary, metric_sd),
            fmt="D",
            color="#222222",
            ecolor="#222222",
            markersize=5,
            capsize=2.5,
            linewidth=1.1,
            zorder=4,
        )


def run_metric(run: TestRun, metric: str) -> float:
    """Return a seed-level value for the requested panel metric."""
    if metric == "pinball_q50":
        return run.pinball_q50
    if metric == "spearman_q50":
        return run.spearman_q50
    if metric == "r2":
        return run.r2_q50_transformed
    raise ValueError(f"Unknown metric {metric!r}")


def summary_metric(summary: TargetSummary, metric: str) -> float:
    """Return a target-level mean for the requested panel metric."""
    if metric == "pinball_q50":
        return summary.pinball_mean
    if metric == "spearman_q50":
        return summary.spearman_mean
    if metric == "r2":
        return summary.r2_mean
    raise ValueError(f"Unknown metric {metric!r}")


def summary_uncertainty(summary: TargetSummary, metric_sd: str) -> float:
    """Return the seed-level sample standard deviation for a panel metric."""
    if metric_sd == "pinball_sd":
        return summary.pinball_sd
    if metric_sd == "spearman_sd":
        return summary.spearman_sd
    if metric_sd == "r2_sd":
        return summary.r2_sd
    raise ValueError(f"Unknown uncertainty field {metric_sd!r}")


def make_legend_handles() -> list[Line2D]:
    """Create a legend for seeds and the mean with seed standard deviation."""
    handles = [
        Line2D(
            [0], [0], marker=SEED_MARKERS[seed], color="none",
            markerfacecolor=SEED_COLORS[seed], markersize=6,
            label=f"Seed {seed}",
        )
        for seed in sorted(SEED_MARKERS)
    ]
    handles.append(
        Line2D(
            [0], [0], marker="D", color="#222222", markersize=5,
            linewidth=1.1, label="Mean ± SD",
        )
    )
    return handles


def create_figure(summaries: list[TargetSummary], output_dir: Path) -> list[Path]:
    """Create three aligned test-metric panels and save PNG and PDF files."""
    configure_plot_style()
    figure, raw_axes = cast(
        tuple[Figure, Iterable[Axes]],
        plt.subplots(
        ncols=3,
        figsize=(17, 9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
        ),
    )
    axes = tuple(raw_axes)
    panels = [
        (axes[0], "pinball_q50", "pinball_sd", "Test q50 pinball loss ↓", "Value (transformed)"),
        (axes[1], "spearman_q50", "spearman_sd", "Test Spearman ρ ↑", "Correlation"),
        (axes[2], "r2", "r2_sd", "Test R² q50 ↑", "R² (transformed)"),
    ]
    y_positions = list(range(len(summaries)))
    labels = [
        f"{TARGET_LABELS[item.target]}  (n={item.sample_count:,})"
        for item in summaries
    ]
    for axis, metric, metric_sd, title, direction in panels:
        draw_seed_points(axis, summaries, metric, metric_sd)
        _ = axis.set_title(title)
        _ = axis.set_xlabel(direction)
        axis.grid(axis="x", color="#D9DDE2", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    _ = axes[0].set_yticks(y_positions, labels)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    axes[0].invert_yaxis()
    _ = axes[0].set_xlim(left=0)
    _ = axes[1].set_xlim(-1, 1)
    r2_low = min(item.r2_mean - item.r2_sd for item in summaries)
    _ = axes[2].set_xlim(min(0.0, r2_low - 0.05), 1.0)
    _ = axes[2].axvline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    _ = figure.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    _ = figure.suptitle(
        "Held-out test results · checkpoints selected on validation", y=0.985, fontsize=11
    )
    figure.subplots_adjust(left=0.32, right=0.99, top=0.92, bottom=0.12, wspace=0.28)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "ten_target_test_metrics.png",
        output_dir / "ten_target_test_metrics.pdf",
    ]
    for output_path in output_paths:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_paths


def parse_args() -> PlotArguments:
    """Parse input and output paths plus the expected seed count."""
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    arguments = PlotArguments(
        input_csv=root / "outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv",
        output_dir=root / "outputs/xrs_only/v1_test_evaluation/figures",
        expected_seeds=3,
    )
    _ = parser.add_argument(
        "--input-csv",
        type=Path,
        default=root / "outputs/xrs_only/v1_test_evaluation/summary/test_metrics_summary.csv",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/xrs_only/v1_test_evaluation/figures",
    )
    _ = parser.add_argument("--expected-seeds", type=int, default=3)
    _ = parser.parse_args(namespace=arguments)
    return arguments


def main() -> None:
    """Load final metrics, validate coverage, and write figure files."""
    args = parse_args()
    runs = load_test_runs(args.input_csv, args.expected_seeds)
    summaries = summarize_runs(runs)
    output_paths = create_figure(summaries, args.output_dir)
    for output_path in output_paths:
        logger.info("Saved test metrics figure: {}", output_path)


if __name__ == "__main__":
    main()
