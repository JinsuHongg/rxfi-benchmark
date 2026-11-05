"""Plot operational binary decision metrics (TSS, POD, HSS, F1+) for Project 1."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger
from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSeriesConfig:
    """Mapping of target names to display aesthetics."""

    target_name: str
    display_label: str
    bar_color: str


DEFAULT_SERIES_CONFIGS: tuple[TargetSeriesConfig, ...] = (
    TargetSeriesConfig(
        target_name="cumulative_peak_flux",
        display_label="Cumulative Peak Flux (Proposed)",
        bar_color="#1D4ED8",
    ),
    TargetSeriesConfig(
        target_name="max_peak_flux",
        display_label="Max Peak Flux (Conventional)",
        bar_color="#059669",
    ),
    TargetSeriesConfig(
        target_name="max_rxfi",
        display_label="Max RXFI",
        bar_color="#DC2626",
    ),
    TargetSeriesConfig(
        target_name="cumulative_rxfi",
        display_label="Cumulative RXFI",
        bar_color="#6B7280",
    ),
)

METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("TSS", "TSS\n(Space Weather Skill)"),
    ("HSS", "HSS\n(Skill vs. Chance)"),
    ("F1_positive", "Positive F1\n(Precision-Recall Balance)"),
    ("POD", "POD (Recall)\n(Detection Rate)"),
)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options for binary metric figure generation.

    Returns:
        Populated arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot operational binary decision metrics for Project 1."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/severe_event_binary_metrics/project1_qr_test_summary.csv"),
        help="Path to Project 1 QR test summary metrics CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/slide_figures/figure_project1_binary_decision_metrics.png"
        ),
        help="Output destination path for the saved figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution DPI for slide presentation export.",
    )
    return parser.parse_args()


def load_metric_table(summary_csv: Path) -> pd.DataFrame:
    """Load and filter summary metrics specifically for M+ severe-flare events.

    Args:
        summary_csv: Path to input evaluation summary table.

    Returns:
        Filtered DataFrame keyed by target.
    """
    if not summary_csv.exists():
        msg = f"Summary CSV not found: {summary_csv}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(summary_csv)
    # Project 1 focus is the M+ forecasting horizon
    m_plus_df = df[df["event_definition"] == "M+"].copy()
    return m_plus_df.set_index("target")


def configure_matplotlib_styling() -> None:
    """Configure plot typography and layout aesthetics for slide viewing."""
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def add_value_annotations(
    ax: plt.Axes,
    bar_containers: Sequence[BarContainer],
) -> None:
    """Annotate numeric values above each individual bar.

    Args:
        ax: Axes containing bar plots.
        bar_containers: Collection of matplotlib bar containers.
    """
    for container in bar_containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                rotation=0,
            )


def render_grouped_bar_chart(
    metrics_by_target: pd.DataFrame,
    series_configs: Sequence[TargetSeriesConfig],
    output_path: Path,
    dpi: int,
) -> None:
    """Generate and save grouped bar chart comparing binary operational metrics.

    Args:
        metrics_by_target: Indexed DataFrame containing target metrics.
        series_configs: Configurations governing display label and bar color.
        output_path: Destination path for output PNG.
        dpi: Output image resolution.
    """
    configure_matplotlib_styling()
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=dpi)

    num_metrics = len(METRIC_COLUMNS)
    num_targets = len(series_configs)
    cluster_indices = np.arange(num_metrics)

    # Bar geometry ensuring compact and distinct clusters
    total_cluster_width = 0.72
    individual_bar_width = total_cluster_width / num_targets

    bar_containers = []
    for idx, cfg in enumerate(series_configs):
        offset = (idx - (num_targets - 1) / 2.0) * individual_bar_width
        metric_values = [
            float(metrics_by_target.loc[cfg.target_name, col_key])
            for col_key, _ in METRIC_COLUMNS
        ]
        container = ax.bar(
            cluster_indices + offset,
            metric_values,
            width=individual_bar_width,
            label=cfg.display_label,
            color=cfg.bar_color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bar_containers.append(container)

    add_value_annotations(ax, bar_containers)

    ax.set_title(
        "Operational Binary Decision Metrics for M+ Flares (Test Set)",
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(cluster_indices)
    ax.set_xticklabels([label for _, label in METRIC_COLUMNS], fontweight="semibold")
    ax.set_ylabel("Metric Score")
    ax.set_ylim([0.0, 1.12])
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    ax.legend(
        loc="upper left",
        ncol=2,
        framealpha=0.95,
        edgecolor="#D1D5DB",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.success(f"Operational binary metrics plot saved to {output_path}")


def main() -> None:
    """Coordinate reading, plotting, and file export."""
    args = parse_arguments()
    logger.info("Starting Project 1 operational decision metrics plot generation.")
    metrics_table = load_metric_table(args.summary_csv)
    render_grouped_bar_chart(
        metrics_by_target=metrics_table,
        series_configs=DEFAULT_SERIES_CONFIGS,
        output_path=args.output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
