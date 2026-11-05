"""Plot Project 2 quantile regression metrics across input channel conditions."""

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
class ChannelDisplayConfig:
    """Mapping of channel identifier to presentation properties."""

    name: str
    display_label: str
    bar_color: str


DEFAULT_CHANNELS: tuple[ChannelDisplayConfig, ...] = (
    ChannelDisplayConfig(
        name="all13",
        display_label="All 13 Channels (Best)",
        bar_color="#059669",
    ),
    ChannelDisplayConfig(
        name="aia131",
        display_label="AIA 131",
        bar_color="#1D4ED8",
    ),
    ChannelDisplayConfig(
        name="hmi_m",
        display_label="HMI_M (Magnetic)",
        bar_color="#6B7280",
    ),
    ChannelDisplayConfig(
        name="aia193",
        display_label="AIA 193",
        bar_color="#D97706",
    ),
)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options for figure generation.

    Returns:
        Populated arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate Project 2 quantile regression evaluation figure."
    )
    parser.add_argument(
        "--continuous-csv",
        type=Path,
        default=Path("outputs/project2/table_qr_cumulative_peak_flux_test.csv"),
        help="Path to continuous quantile regression test metrics CSV.",
    )
    parser.add_argument(
        "--binary-csv",
        type=Path,
        default=Path(
            "outputs/severe_event_binary_metrics/project2_qr_test_summary.csv"
        ),
        help="Path to binary decision metrics summary CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/slide_figures/figure_project2_qr_metrics.png"),
        help="Target file path to save the generated figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution DPI for exported presentation figure.",
    )
    return parser.parse_args()


def load_qr_metrics(continuous_csv: Path, binary_csv: Path) -> pd.DataFrame:
    """Load and merge continuous regression and binary M+ alert metrics.

    Args:
        continuous_csv: CSV containing Spearman, R2, and Normalized MAE.
        binary_csv: CSV containing M+ operational TSS and HSS.

    Returns:
        Consolidated DataFrame indexed by input channel identifier.
    """
    if not continuous_csv.exists():
        msg = f"Continuous metrics CSV missing: {continuous_csv}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not binary_csv.exists():
        msg = f"Binary metrics CSV missing: {binary_csv}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df_cont = pd.read_csv(continuous_csv).set_index("input")
    df_bin = pd.read_csv(binary_csv)
    df_bin_m = df_bin[
        (df_bin["event_definition"] == "M+") & (df_bin["evaluation_split"] == "test")
    ].set_index("input_condition")

    combined = pd.DataFrame(index=df_cont.index)
    combined["spearman"] = df_cont["spearman"]
    combined["r2"] = df_cont["r2"]
    combined["mae_iqr"] = df_cont["mae_iqr"]
    combined["tss"] = df_bin_m["TSS"]
    combined["hss"] = df_bin_m["HSS"]
    return combined


def configure_matplotlib_fonts() -> None:
    """Set typographic styling optimized for distant presentation readability."""
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def annotate_bar_heights(ax: plt.Axes, containers: Sequence[BarContainer]) -> None:
    """Print numeric value labels with angled orientation to prevent collisions.

    Args:
        ax: Subplot axes.
        containers: Sequence of matplotlib bar containers.
    """
    for container in containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                rotation=35,
            )


def render_metric_panel(
    ax: plt.Axes,
    metric_names: Sequence[str],
    metric_labels: Sequence[str],
    data_df: pd.DataFrame,
    channel_configs: Sequence[ChannelDisplayConfig],
    title: str,
    y_limit: float,
) -> None:
    """Render a grouped cluster bar chart on designated subplot axes.

    Args:
        ax: Matplotlib target axes.
        metric_names: Column keys from DataFrame to plot.
        metric_labels: Display labels for x-axis ticks.
        data_df: DataFrame indexed by channel identifier.
        channel_configs: Configurations governing channels, colors, and labels.
        title: Title string for the panel.
        y_limit: Upper limit for y-axis.
    """
    num_metrics = len(metric_names)
    num_channels = len(channel_configs)
    cluster_indices = np.arange(num_metrics)

    cluster_width = 0.76
    bar_width = cluster_width / num_channels

    bar_containers = []
    for idx, cfg in enumerate(channel_configs):
        offset = (idx - (num_channels - 1) / 2.0) * bar_width
        vals = [float(data_df.loc[cfg.name, col]) for col in metric_names]
        container = ax.bar(
            cluster_indices + offset,
            vals,
            width=bar_width,
            label=cfg.display_label,
            color=cfg.bar_color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bar_containers.append(container)

    annotate_bar_heights(ax, bar_containers)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(cluster_indices)
    ax.set_xticklabels(metric_labels, fontweight="semibold")
    ax.set_ylabel("Metric Score")
    ax.set_ylim([0.0, y_limit])
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#D1D5DB")


def generate_qr_figure(
    data_df: pd.DataFrame,
    channel_configs: Sequence[ChannelDisplayConfig],
    output_path: Path,
    dpi: int,
) -> None:
    """Generate and save dual-panel quantile regression comparison figure.

    Args:
        data_df: Metrics table containing evaluation values.
        channel_configs: Channel display specifications.
        output_path: Output file destination.
        dpi: Dots per inch resolution.
    """
    configure_matplotlib_fonts()
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 6.2),
        dpi=dpi,
        gridspec_kw={"width_ratios": [1.32, 1.0]},
    )

    render_metric_panel(
        ax=ax1,
        metric_names=("spearman", "r2", "mae_iqr"),
        metric_labels=("Spearman ρ", "R²", "MAE (Normalized)"),
        data_df=data_df,
        channel_configs=channel_configs,
        title="Continuous Learnability & Regression Quality",
        y_limit=1.28,
    )

    render_metric_panel(
        ax=ax2,
        metric_names=("tss", "hss"),
        metric_labels=("TSS", "HSS"),
        data_df=data_df,
        channel_configs=channel_configs,
        title="Operational M+ Alert Skill",
        y_limit=0.82,
    )

    plt.tight_layout(w_pad=3.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.success(f"Project 2 QR figure saved to {output_path}")


def main() -> None:
    """Coordinate reading, plotting, and file export."""
    args = parse_arguments()
    logger.info("Initiating Project 2 quantile regression plot generation.")
    metrics_table = load_qr_metrics(
        continuous_csv=args.continuous_csv,
        binary_csv=args.binary_csv,
    )
    generate_qr_figure(
        data_df=metrics_table,
        channel_configs=DEFAULT_CHANNELS,
        output_path=args.output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
