"""Plot Precision-Recall and ROC curves for Project 1 severe-flare forecasts."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


@dataclass(frozen=True)
class TargetDisplayConfig:
    """Configuration mapping target files to visual presentation properties."""

    name: str
    display_label: str
    file_path: Path
    line_color: str
    display_zorder: int


DEFAULT_TARGET_CONFIGS: tuple[TargetDisplayConfig, ...] = (
    TargetDisplayConfig(
        name="cumulative_peak_flux",
        display_label="Cumulative Peak Flux",
        file_path=Path("outputs/project3_exports/cumulative_peak_flux/test.csv"),
        line_color="#1D4ED8",
        display_zorder=4,
    ),
    TargetDisplayConfig(
        name="max_peak_flux",
        display_label="Max Peak Flux (Conventional)",
        file_path=Path("outputs/project3_exports/max_peak_flux/test.csv"),
        line_color="#059669",
        display_zorder=3,
    ),
    TargetDisplayConfig(
        name="max_rxfi",
        display_label="Max RXFI",
        file_path=Path("outputs/project3_exports/max_rxfi/test.csv"),
        line_color="#DC2626",
        display_zorder=2,
    ),
    TargetDisplayConfig(
        name="cumulative_rxfi",
        display_label="Cumulative RXFI",
        file_path=Path("outputs/project3_exports/cumulative_rxfi/test.csv"),
        line_color="#6B7280",
        display_zorder=1,
    ),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for figure generation.

    Returns:
        Parsed arguments container.
    """
    parser = argparse.ArgumentParser(
        description="Generate PR and ROC curves for Project 1 M+ flare prediction."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/slide_figures/figure_project1_m_plus_curves.png"),
        help="Target file path to save the generated figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for slide graphics.",
    )
    return parser.parse_args()


def load_binary_eval_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load continuous q50 predictions and construct binary severe-flare labels.

    Args:
        file_path: Location of the test prediction export CSV.

    Returns:
        Tuple of (ground_truth_binary, predicted_continuous_scores).
    """
    if not file_path.exists():
        msg = f"Evaluation prediction artifact not found: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(file_path)
    # M+ outcomes correspond to ordinal flare bands M and X (index 4 and 5)
    y_true = (df["ordinal_class_index"] >= 4).to_numpy(dtype=int)
    y_score = df["q50"].to_numpy(dtype=float)
    return y_true, y_score


def configure_matplotlib_fonts() -> None:
    """Set presentation font and line sizes for distant legibility."""
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "lines.linewidth": 2.6,
        }
    )


def style_precision_recall_panel(ax: plt.Axes, prevalence: float) -> None:
    """Apply styling, labels, and reference baseline to PR panel.

    Args:
        ax: Target matplotlib axes for precision-recall curves.
        prevalence: Positive class proportion serving as no-skill baseline.
    """
    ax.axhline(
        y=prevalence,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.8,
        label=f"No-skill Baseline ({prevalence:.3f})",
    )
    ax.set_title("Precision-Recall Curve (M+ Flares)", fontweight="bold")
    ax.set_xlabel("Recall (Probability of Detection)")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.92)


def style_roc_panel(ax: plt.Axes) -> None:
    """Apply styling, labels, and diagonal chance line to ROC panel.

    Args:
        ax: Target matplotlib axes for ROC curves.
    """
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.8,
        label="Chance Line (0.500)",
    )
    ax.set_title("ROC Curve (M+ Flares)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (POFD)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.92)


def generate_m_plus_curves(
    target_configs: Sequence[TargetDisplayConfig],
    output_path: Path,
    dpi: int,
) -> None:
    """Generate and save dual-panel PR and ROC evaluation figure.

    Args:
        target_configs: Sequence of target configurations to evaluate.
        output_path: Output file destination.
        dpi: Dot resolution per inch for export.
    """
    configure_matplotlib_fonts()
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(15, 6), dpi=dpi)

    calculated_prevalence: float = 0.0

    for cfg in target_configs:
        logger.info(f"Processing evaluation curves for {cfg.name}...")
        y_true, y_score = load_binary_eval_data(cfg.file_path)
        calculated_prevalence = float(np.mean(y_true))

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_score = auc(recall, precision)
        ax_pr.plot(
            recall,
            precision,
            label=f"{cfg.display_label} (AUPRC = {pr_score:.3f})",
            color=cfg.line_color,
            zorder=cfg.display_zorder,
        )

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_score = auc(fpr, tpr)
        ax_roc.plot(
            fpr,
            tpr,
            label=f"{cfg.display_label} (AUROC = {roc_score:.3f})",
            color=cfg.line_color,
            zorder=cfg.display_zorder,
        )

    style_precision_recall_panel(ax_pr, calculated_prevalence)
    style_roc_panel(ax_roc)

    plt.tight_layout(w_pad=3.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.success(f"Generated M+ curves saved to {output_path}")


def main() -> None:
    """Coordinate script execution and figure persistence."""
    args = parse_arguments()
    logger.info("Initiating Project 1 M+ curve plotting workflow.")
    generate_m_plus_curves(
        target_configs=DEFAULT_TARGET_CONFIGS,
        output_path=args.output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
