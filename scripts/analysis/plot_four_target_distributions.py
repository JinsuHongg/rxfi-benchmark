"""Plot training-only histograms for four selected XRS targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.plot_v1_target_distributions import (
    DEFAULT_STATS_JSON,
    DEFAULT_TRAIN_CSV,
    FAMILY_COLORS,
    StatsDocument,
    TARGET_UNITS,
    configure_plot_style,
    load_stats,
    matched_rows,
    sha256_file,
    summarize_targets,
    target_family,
)
from scripts.training.compute_target_stats import DEFAULT_ZARR_PATH

TARGET_ORDER = (
    "max_xflux",
    "fluence",
    "max_background_deviation_24h_q50",
    "total_background_deviation_24h_q50",
)
TARGET_LABELS = {
    "max_xflux": "Max X-ray flux",
    "fluence": "Fluence",
    "max_background_deviation_24h_q50": "Max BS X-ray flux (q50, 24 h)",
    "total_background_deviation_24h_q50": "Cumulative BS X-ray flux (q50, 24 h)",
}
DEFAULT_OUTPUT_DIR = Path("outputs/xrs_only/v1_four_target_distributions")
DEFAULT_HISTOGRAM_BINS = 80
FloatArray = NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for the four-target histogram."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    return parser.parse_args()


def display_path(path: Path) -> str:
    """Use a repository-relative path when possible for portable metadata."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def draw_histogram(axis: Axes, target: str, values: FloatArray, edges: FloatArray) -> None:
    """Draw one density-normalized histogram using shared transformed bins."""
    _ = axis.hist(
        values,
        bins=cast(list[float], edges.tolist()),
        density=True,
        color=FAMILY_COLORS[target_family(target)],
        edgecolor="white",
        linewidth=0.25,
    )
    _ = axis.set_title(TARGET_LABELS[target], loc="left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def save_histogram(
    output_dir: Path,
    distributions: dict[str, FloatArray],
    bin_count: int,
) -> tuple[Path, Path]:
    """Save the four-panel shared-bin histogram as PNG and PDF."""
    if bin_count < 10:
        raise ValueError("At least ten histogram bins are required")
    edges = np.linspace(
        min(float(np.min(values)) for values in distributions.values()),
        max(float(np.max(values)) for values in distributions.values()),
        bin_count + 1,
    )

    configure_plot_style()
    figure, raw_axes = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
    axes = cast(list[Axes], raw_axes.flat)
    for axis, target in zip(axes, TARGET_ORDER, strict=True):
        draw_histogram(axis, target, distributions[target], edges)
    _ = figure.supxlabel(r"Scaled signed transform, $\mathrm{asinh}(y/s)$", y=0.02)
    _ = figure.supylabel("Probability density")
    _ = figure.tight_layout(rect=(0.035, 0.035, 1, 1), h_pad=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "four_target_train_distributions_asinh_histogram.png"
    pdf_path = output_dir / "four_target_train_distributions_asinh_histogram.pdf"
    _ = figure.savefig(pdf_path)
    _ = figure.savefig(png_path, dpi=300, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def write_metadata(
    output_dir: Path,
    train_csv: Path,
    zarr_path: Path,
    stats_json: Path,
    stats: StatsDocument,
    zarr_hashes: dict[str, str],
    rows_matched: int,
    bin_count: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Write input provenance and per-target cohort counts beside the plot."""
    target_counts = {name: stats.target_valid_rows[name] for name in TARGET_ORDER}
    metadata = {
        "fit_split": "train",
        "train_csv": display_path(train_csv),
        "train_csv_sha256": stats.source_csv_sha256,
        "stats_json": display_path(stats_json),
        "stats_json_sha256": sha256_file(stats_json),
        "zarr_path": display_path(zarr_path),
        "zarr_time_sha256_by_year": zarr_hashes,
        "cohort_rule": stats.fit_cohort,
        "cohort_counts": {
            "train_rows": stats.train_rows,
            "input_timestamp_matches": rows_matched,
            "missing_input_timestamp_rows": stats.train_rows - rows_matched,
            "target_valid_rows": target_counts,
            "target_missing_rows": {
                name: rows_matched - count for name, count in target_counts.items()
            },
        },
        "targets": [
            {
                "column": name,
                "label": TARGET_LABELS[name],
                "units": TARGET_UNITS[name],
                "asinh_scale_s": stats.targets[name].scale_s,
            }
            for name in TARGET_ORDER
        ],
        "plot": "4-row by 1-column density histogram of asinh(y / s)",
        "histogram_bins": bin_count,
        "histogram_bin_edge_rule": "shared equal-width bins spanning all four transformed target distributions",
        "png": display_path(png_path),
        "pdf": display_path(pdf_path),
        "test_or_validation_data_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Generate the four-target train distribution histogram."""
    args = parse_args()
    if args.histogram_bins < 10:
        raise ValueError("At least ten histogram bins are required")

    stats = load_stats(args.stats_json)
    rows, zarr_hashes = matched_rows(args.train_csv, args.zarr_path, stats)
    all_distributions, _ = summarize_targets(rows, stats)
    distributions = {name: all_distributions[name] for name in TARGET_ORDER}
    png_path, pdf_path = save_histogram(
        args.output_dir, distributions, args.histogram_bins
    )
    write_metadata(
        args.output_dir,
        args.train_csv,
        args.zarr_path,
        args.stats_json,
        stats,
        zarr_hashes,
        len(rows),
        args.histogram_bins,
        png_path,
        pdf_path,
    )
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
