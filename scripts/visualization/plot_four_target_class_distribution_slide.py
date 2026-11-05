#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate a high-contrast publication and slide figure comparing four solar flare targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.use("Agg")

CLASS_ORDER: Final[tuple[str, ...]] = ("FQ", "B", "C", "M", "X")
CLASS_COLORS: Final[dict[str, str]] = {
    "FQ": "#7F8C8D",
    "B": "#74B9FF",
    "C": "#0984E3",
    "M": "#E17055",
    "X": "#D63031",
}
FLUX_SCALE: Final[float] = 1.0e-8


def configure_slide_style() -> None:
    """Set global matplotlib parameters tailored for high-legibility slide presentations."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "legend.title_fontsize": 16,
            "axes.linewidth": 1.8,
            "xtick.major.size": 6,
            "xtick.major.width": 1.5,
            "ytick.major.size": 6,
            "ytick.major.width": 1.5,
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load target dataset and filter records with valid label definitions.

    Args:
        path: Path to the target CSV file.

    Returns:
        Validated pandas DataFrame containing the target columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Target dataset file not found: {path}")

    required_columns = {
        "max_flare_class",
        "max_peak_flux",
        "cumulative_peak_flux",
        "max_rxfi",
        "cumulative_rxfi",
    }
    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Ensure class labels are standard uppercase and map sub-classes to primary bands
    df["clean_class"] = df["max_flare_class"].astype(str).str.strip().str.upper()
    df["clean_class"] = df["clean_class"].apply(
        lambda val: val if val == "FQ" else val[:1]
    )
    return df


def plot_stacked_distribution(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    transform_func: Callable[[np.ndarray], np.ndarray],
    bin_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    """Render a stacked histogram broken down by conventional GOES flare classes.

    Args:
        ax: Target matplotlib axes.
        df: Input DataFrame.
        column: Column name to plot.
        transform_func: Mapping applied to raw values prior to binning.
        bin_edges: 1D array of histogram bin edges.
        xlabel: Explicit X-axis title with scale and units.
        ylabel: Explicit Y-axis title.
    """
    class_groups = [
        transform_func(
            df.loc[df["clean_class"] == flare_class, column].to_numpy(dtype=float)
        )
        for flare_class in CLASS_ORDER
    ]
    colors = [CLASS_COLORS[flare_class] for flare_class in CLASS_ORDER]

    ax.hist(
        class_groups,
        bins=bin_edges,
        stacked=True,
        color=colors,
        label=CLASS_ORDER,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_flux_accumulation_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot max peak flux versus cumulative peak flux to demonstrate multi-flare clustering.

    Args:
        ax: Target matplotlib axes.
        df: Input DataFrame.
        xlabel: Explicit X-axis title.
        ylabel: Explicit Y-axis title.
    """
    # Non-quiet events only to inspect active solar regimes on log-log coordinates
    active = df[df["clean_class"] != "FQ"].copy()
    x = np.log10(active["max_peak_flux"].to_numpy(dtype=float) / FLUX_SCALE)
    y = np.log10(active["cumulative_peak_flux"].to_numpy(dtype=float) / FLUX_SCALE)

    for flare_class in ("B", "C", "M", "X"):
        mask = active["clean_class"] == flare_class
        ax.scatter(
            x[mask],
            y[mask],
            s=12,
            alpha=0.35,
            color=CLASS_COLORS[flare_class],
            label=flare_class,
            rasterized=True,
        )

    # Reference diagonal where cumulative equals maximum (single isolated event)
    min_val, max_val = 0.5, 6.0
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#2D3436",
        linestyle="--",
        linewidth=2.0,
        label="Single Flare (y = x)",
    )
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 6.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=14)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_rxfi_decoupling_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot max peak flux versus max RXFI to expose background-relative decoupling.

    Args:
        ax: Target matplotlib axes.
        df: Input DataFrame.
        xlabel: Explicit X-axis title.
        ylabel: Explicit Y-axis title.
    """
    active = df[df["clean_class"] != "FQ"].copy()
    x = np.log10(active["max_peak_flux"].to_numpy(dtype=float) / FLUX_SCALE)
    y = np.log10(1.0 + active["max_rxfi"].to_numpy(dtype=float))

    for flare_class in ("B", "C", "M", "X"):
        mask = active["clean_class"] == flare_class
        ax.scatter(
            x[mask],
            y[mask],
            s=12,
            alpha=0.35,
            color=CLASS_COLORS[flare_class],
            label=flare_class,
            rasterized=True,
        )

    # Vertical thresholds demarcating conventional GOES C (1e-6), M (1e-5), and X (1e-4) boundaries
    for boundary in (2.0, 3.0, 4.0):
        ax.axvline(boundary, color="#B2BEC3", linestyle=":", linewidth=1.5)

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.0, 3.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)


def render_comparison_slide(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Construct the complete 2x3 scientific comparison figure and write to disk.

    Args:
        df: Prepared targets DataFrame.
        output_path: Destination path for the saved image.
        dpi: Output resolution dots-per-inch.
    """
    configure_slide_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.03, wspace=0.30, hspace=0.04)

    plot_stacked_distribution(
        ax=axes[0, 0],
        df=df,
        column="max_peak_flux",
        transform_func=lambda val: np.log10(val / FLUX_SCALE),
        bin_edges=np.linspace(0.0, 5.5, 23),
        xlabel=r"$\mathbf{Max\ Peak\ Flux\ (\log_{10}[F/10^{-8}\ W\ m^{-2}])}$",
        ylabel=r"$\mathbf{Sample\ Count}$",
    )

    cum_flux_bins = np.linspace(0.0, 6.0, 25)
    plot_stacked_distribution(
        ax=axes[0, 1],
        df=df,
        column="cumulative_peak_flux",
        transform_func=lambda val: np.log10(1.0 + val / FLUX_SCALE),
        bin_edges=cum_flux_bins,
        xlabel=r"$\mathbf{Cumulative\ Peak\ Flux\ (\log_{10}[1 + F/10^{-8}\ W\ m^{-2}])}$",
        ylabel=r"$\mathbf{Sample\ Count}$",
    )

    rxfi_bins = np.linspace(0.0, 3.0, 25)
    plot_stacked_distribution(
        ax=axes[1, 0],
        df=df,
        column="max_rxfi",
        transform_func=lambda val: np.log10(1.0 + val),
        bin_edges=rxfi_bins,
        xlabel=r"$\mathbf{\log_{10}(1 + Max\ RXFI)}$",
        ylabel=r"$\mathbf{Sample\ Count}$",
    )

    plot_stacked_distribution(
        ax=axes[1, 1],
        df=df,
        column="cumulative_rxfi",
        transform_func=lambda val: np.log10(1.0 + val),
        bin_edges=rxfi_bins,
        xlabel=r"$\mathbf{\log_{10}(1 + Cumulative\ RXFI)}$",
        ylabel=r"$\mathbf{Sample\ Count}$",
    )

    plot_flux_accumulation_scatter(
        ax=axes[0, 2],
        df=df,
        xlabel=r"$\mathbf{Max\ Peak\ Flux\ (\log_{10}[F/10^{-8}\ W\ m^{-2}])}$",
        ylabel=r"$\mathbf{Cumulative\ Peak\ Flux}$"
        + "\n"
        + r"$\mathbf{(\log_{10}[F/10^{-8}\ W\ m^{-2}])}$",
    )

    plot_rxfi_decoupling_scatter(
        ax=axes[1, 2],
        df=df,
        xlabel=r"$\mathbf{Max\ Peak\ Flux\ (\log_{10}[F/10^{-8}\ W\ m^{-2}])}$",
        ylabel=r"$\mathbf{\log_{10}(1 + Max\ RXFI)}$",
    )

    axes[0, 0].legend(title="Max Class", loc="upper right", frameon=True)
    axes[0, 2].legend(loc="lower right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Slide figure saved successfully: {}", output_path)


def main() -> None:
    """Parse CLI options and execute slide figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate 4-label comparison figure for presentation slides."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/derived/test_targets.csv"),
        help="Path to the derived split target CSV (default: data/derived/test_targets.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/slide_figures/figure_four_target_distribution_slide.png"),
        help="Path where output image will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution DPI for slide presentation.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_csv)
    render_comparison_slide(dataset, args.output_path, args.dpi)


if __name__ == "__main__":
    main()
