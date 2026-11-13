# pyright: reportUnknownMemberType=false, reportAny=false
"""Plot training-only distributions for the ten V1 XRS targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.axes import Axes
from numpy.typing import NDArray

from scripts.training.compute_target_stats import (
    DEFAULT_ZARR_PATH,
    TARGET_COLUMNS,
    TrainingRow,
    read_training_rows,
    read_zarr_times,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET_ORDER = (
    "max_xflux",
    "fluence",
    "max_background_deviation_24h_q10",
    "max_background_deviation_24h_q50",
    "max_background_deviation_2h_q10",
    "max_background_deviation_2h_q50",
    "total_background_deviation_24h_q10",
    "total_background_deviation_24h_q50",
    "total_background_deviation_2h_q10",
    "total_background_deviation_2h_q50",
)
TARGET_LABELS = {
    "max_xflux": "Maximum X-ray flux",
    "fluence": "Fluence",
    "max_background_deviation_24h_q10": "Max background deviation · 24 h · q10",
    "max_background_deviation_24h_q50": "Max background deviation · 24 h · q50",
    "max_background_deviation_2h_q10": "Max background deviation · 2 h · q10",
    "max_background_deviation_2h_q50": "Max background deviation · 2 h · q50",
    "total_background_deviation_24h_q10": "Total background deviation · 24 h · q10",
    "total_background_deviation_24h_q50": "Total background deviation · 24 h · q50",
    "total_background_deviation_2h_q10": "Total background deviation · 2 h · q10",
    "total_background_deviation_2h_q50": "Total background deviation · 2 h · q50",
}
HISTOGRAM_TARGET_LABELS = {
    "max_background_deviation_24h_q10": "Max. BS X-ray flux (24 h, q10)",
    "max_background_deviation_24h_q50": "Max. BS X-ray flux (24 h, q50)",
    "max_background_deviation_2h_q10": "Max. BS X-ray flux (2 h, q10)",
    "max_background_deviation_2h_q50": "Max. BS X-ray flux (2 h, q50)",
    "total_background_deviation_24h_q10": "Cum. BS X-ray flux (24 h, q10)",
    "total_background_deviation_24h_q50": "Cum. BS X-ray flux (24 h, q50)",
    "total_background_deviation_2h_q10": "Cum. BS X-ray flux (2 h, q10)",
    "total_background_deviation_2h_q50": "Cum. BS X-ray flux (2 h, q50)",
}
TARGET_UNITS = {
    "max_xflux": "W m^-2",
    "fluence": "unscaled one-minute sample sum",
    "max_background_deviation_24h_q10": "W m^-2",
    "max_background_deviation_24h_q50": "W m^-2",
    "max_background_deviation_2h_q10": "W m^-2",
    "max_background_deviation_2h_q50": "W m^-2",
    "total_background_deviation_24h_q10": "unscaled one-minute sample sum",
    "total_background_deviation_24h_q50": "unscaled one-minute sample sum",
    "total_background_deviation_2h_q10": "unscaled one-minute sample sum",
    "total_background_deviation_2h_q50": "unscaled one-minute sample sum",
}
FAMILY_COLORS = {
    "max_xflux": "#0072B2",
    "fluence": "#0072B2",
    "max_background_deviation": "#009E73",
    "total_background_deviation": "#D55E00",
}
DEFAULT_TRAIN_CSV = Path("data/derived_noaa_g17_v222/train.csv")
DEFAULT_STATS_JSON = Path("outputs/xrs_only/v1_test_evaluation/summary/train_target_stats.json")
DEFAULT_OUTPUT_DIR = Path("outputs/xrs_only/v1_target_distributions")
DEFAULT_HISTOGRAM_BINS = 80
FIGURE_XLABEL_Y = 0.02
SUBPLOT_BOTTOM_MARGIN = 0.035
SUMMARY_COLUMNS: tuple[str, ...] = (
    "target",
    "units",
    "n_train",
    "n_missing",
    "missing_percent",
    "negative_percent",
    "zero_percent",
    "raw_min",
    "p01",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "raw_max",
    "asinh_scale_s",
)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TargetStat:
    """Store the target-specific training count and asinh scale."""

    n_train: int
    scale_s: float


@dataclass(frozen=True)
class StatsDocument:
    """Hold only validated fields needed from the target stats JSON."""

    source_csv_sha256: str
    zarr_time_hashes: dict[str, str]
    train_rows: int
    matched_rows: int
    target_valid_rows: dict[str, int]
    targets: dict[str, TargetStat]
    fit_cohort: str


@dataclass(frozen=True)
class PlotArgs:
    """Store parsed paths for this analysis."""

    train_csv: Path
    zarr_path: Path
    stats_json: Path
    output_dir: Path
    histogram_bins: int


def parse_args() -> PlotArgs:
    """Parse command-line paths for the V1 training distribution analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    _ = parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    _ = parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    _ = parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument(
        "--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS
    )
    parsed = parser.parse_args()
    return PlotArgs(
        train_csv=cast(Path, parsed.train_csv),
        zarr_path=cast(Path, parsed.zarr_path),
        stats_json=cast(Path, parsed.stats_json),
        output_dir=cast(Path, parsed.output_dir),
        histogram_bins=cast(int, parsed.histogram_bins),
    )


def object_map(value: object, name: str) -> dict[str, object]:
    """Validate a JSON object before reading its named values."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object for {name}")
    return cast(dict[str, object], value)


def integer_value(value: object, name: str) -> int:
    """Validate an integer field from the saved JSON metadata."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected an integer for {name}")
    return value


def float_value(value: object, name: str) -> float:
    """Validate a numeric field from the saved JSON metadata."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected a number for {name}")
    return float(value)


def load_stats(path: Path) -> StatsDocument:
    """Read and validate the saved training statistics metadata."""
    with path.open(encoding="utf-8") as handle:
        payload = object_map(json.load(handle), "root")
    if payload.get("fit_split") != "train":
        raise ValueError(f"Expected train-only statistics in {path}")
    cohort = object_map(payload["cohort"], "cohort")
    raw_counts = object_map(cohort["target_valid_rows"], "target_valid_rows")
    target_counts = {
        name: integer_value(raw_counts[name], f"target_valid_rows.{name}")
        for name in TARGET_ORDER
    }
    raw_target_stats = object_map(payload["targets"], "targets")
    targets = {
        name: TargetStat(
            n_train=integer_value(
                object_map(raw_target_stats[name], name)["n_train"],
                f"targets.{name}.n_train",
            ),
            scale_s=float_value(
                object_map(raw_target_stats[name], name)["scale_s"],
                f"targets.{name}.scale_s",
            ),
        )
        for name in TARGET_ORDER
    }
    raw_hashes = object_map(payload["zarr_time_sha256_by_year"], "zarr_time hashes")
    hashes = {str(year): str(value) for year, value in raw_hashes.items()}
    return StatsDocument(
        source_csv_sha256=str(payload["source_csv_sha256"]),
        zarr_time_hashes=hashes,
        train_rows=integer_value(cohort["train_rows"], "cohort.train_rows"),
        matched_rows=integer_value(
            cohort["input_timestamp_matches"], "cohort.input_timestamp_matches"
        ),
        target_valid_rows=target_counts,
        targets=targets,
        fit_cohort=str(payload["fit_cohort"]),
    )


def matched_rows(
    train_csv: Path, zarr_path: Path, stats: StatsDocument
) -> tuple[list[TrainingRow], dict[str, str]]:
    """Return finite-target rows whose timestamps exist in the V1 Zarr index."""
    actual_csv_hash = sha256_file(train_csv)
    if actual_csv_hash != stats.source_csv_sha256:
        raise ValueError("V1 train CSV hash does not match the target statistics")
    rows = read_training_rows(train_csv)
    times_by_year, hashes_by_year = read_zarr_times(
        zarr_path, {row.year for row in rows}
    )
    if hashes_by_year != stats.zarr_time_hashes:
        raise ValueError("Zarr time-index hashes differ from the saved target statistics")
    matched = [
        row for row in rows if row.timestamp_ns in times_by_year.get(row.year, set())
    ]
    if len(rows) != stats.train_rows:
        raise ValueError("V1 train row count differs from the saved target statistics")
    if len(matched) != stats.matched_rows:
        raise ValueError("Zarr-matched train row count differs from the saved statistics")
    return matched, hashes_by_year


def summarize_targets(
    rows: list[TrainingRow], stats: StatsDocument
) -> tuple[dict[str, FloatArray], list[dict[str, str | int | float]]]:
    """Calculate raw-unit summaries and transformed values per target."""
    distributions: dict[str, FloatArray] = {}
    summaries: list[dict[str, str | int | float]] = []
    expected_counts = stats.target_valid_rows
    for target in TARGET_ORDER:
        raw_values = np.asarray(
            [row.targets[target] for row in rows if row.targets[target] is not None],
            dtype=np.float64,
        )
        target_stats = stats.targets[target]
        if raw_values.size != expected_counts[target] or raw_values.size != target_stats.n_train:
            raise ValueError(f"Training target count differs for {target}")
        scale = target_stats.scale_s
        distributions[target] = np.arcsinh(raw_values / scale)
        percentiles = np.percentile(raw_values, [1, 25, 50, 75, 95, 99])
        missing = len(rows) - raw_values.size
        summaries.append(
            {
                "target": target,
                "units": TARGET_UNITS[target],
                "n_train": int(raw_values.size),
                "n_missing": int(missing),
                "missing_percent": 100.0 * missing / len(rows),
                "negative_percent": 100.0 * np.count_nonzero(raw_values < 0) / raw_values.size,
                "zero_percent": 100.0 * np.count_nonzero(raw_values == 0) / raw_values.size,
                "raw_min": float(np.min(raw_values)),
                "p01": float(percentiles[0]),
                "p25": float(percentiles[1]),
                "median": float(percentiles[2]),
                "p75": float(percentiles[3]),
                "p95": float(percentiles[4]),
                "p99": float(percentiles[5]),
                "raw_max": float(np.max(raw_values)),
                "asinh_scale_s": scale,
            }
        )
    return distributions, summaries


def configure_plot_style() -> None:
    """Set readable fonts and embedded vector text for manuscript output."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.titlepad": 5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def target_family(target: str) -> str:
    """Return the target-family key used to assign accessible colors."""
    if target.startswith("max_background_deviation"):
        return "max_background_deviation"
    if target.startswith("total_background_deviation"):
        return "total_background_deviation"
    return target


def draw_distribution(axis: Axes, target: str, values: FloatArray) -> None:
    """Draw a full-range empirical CDF for one asinh-scaled target."""
    ordered = np.sort(values)
    cdf = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    _ = axis.step(
        ordered,
        cdf,
        where="post",
        color=FAMILY_COLORS[target_family(target)],
        linewidth=1.25,
    )
    _ = axis.set_title(f"{TARGET_LABELS[target]}  (n={ordered.size:,})", loc="left")
    _ = axis.set_ylim(0, 1)
    _ = axis.set_yticks([0, 0.5, 1])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def save_ecdf_figure(output_dir: Path, distributions: dict[str, FloatArray]) -> None:
    """Save the 10-panel ECDF figure as publication PDF and preview PNG."""
    configure_plot_style()
    figure, raw_axes = plt.subplots(5, 2, figsize=(8.2, 10.5), sharey=True)
    axes = cast(list[Axes], raw_axes.flat)
    for axis, target in zip(axes, TARGET_ORDER, strict=True):
        draw_distribution(axis, target, distributions[target])
    _ = figure.supxlabel(
        r"Scaled signed transform, $\mathrm{asinh}(y/s)$", y=FIGURE_XLABEL_Y
    )
    _ = figure.supylabel("Empirical cumulative fraction")
    _ = figure.tight_layout(
        rect=(0.035, SUBPLOT_BOTTOM_MARGIN, 1, 1), h_pad=1.0, w_pad=0.9
    )
    _ = figure.savefig(output_dir / "v1_target_train_distributions_asinh_ecdf.pdf")
    _ = figure.savefig(
        output_dir / "v1_target_train_distributions_asinh_ecdf.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)


def histogram_edges(
    distributions: dict[str, FloatArray], bin_count: int
) -> FloatArray:
    """Create common bin edges that cover every transformed observation."""
    if bin_count < 10:
        raise ValueError("At least ten histogram bins are required")
    all_values = np.concatenate(
        [distributions[target] for target in TARGET_ORDER]
    )
    return np.linspace(float(np.min(all_values)), float(np.max(all_values)), bin_count + 1)


def draw_histogram(
    axis: Axes, target: str, values: FloatArray, edges: FloatArray
) -> None:
    """Draw a density-normalized histogram for one transformed target."""
    _ = axis.hist(
        values,
        bins=cast(list[float], edges.tolist()),
        density=True,
        color=FAMILY_COLORS[target_family(target)],
        edgecolor="white",
        linewidth=0.25,
    )
    label = HISTOGRAM_TARGET_LABELS.get(target, TARGET_LABELS[target])
    _ = axis.set_title(label, loc="left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def save_histogram_figure(
    output_dir: Path,
    distributions: dict[str, FloatArray],
    bin_count: int,
) -> None:
    """Save common-bin, density-normalized target histograms."""
    configure_plot_style()
    edges = histogram_edges(distributions, bin_count)
    figure, raw_axes = plt.subplots(5, 2, figsize=(8.2, 10.5), sharex=True)
    axes = cast(list[Axes], raw_axes.flat)
    for axis, target in zip(axes, TARGET_ORDER, strict=True):
        draw_histogram(axis, target, distributions[target], edges)
    _ = figure.supxlabel(
        r"Scaled signed transform, $\mathrm{asinh}(y/s)$", y=FIGURE_XLABEL_Y
    )
    _ = figure.supylabel("Probability density")
    _ = figure.tight_layout(
        rect=(0.035, SUBPLOT_BOTTOM_MARGIN, 1, 1), h_pad=1.0, w_pad=0.9
    )
    _ = figure.savefig(output_dir / "v1_target_train_distributions_asinh_histogram.pdf")
    _ = figure.savefig(
        output_dir / "v1_target_train_distributions_asinh_histogram.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)


def write_summary_csv(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write raw-unit quantiles and cohort counts to CSV."""
    path = output_dir / "target_distribution_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SUMMARY_COLUMNS),
            lineterminator="\n",
        )
        _ = writer.writeheader()
        for row in rows:
            _ = writer.writerow(row)


def tex_num(value: float) -> str:
    """Format a finite value in notation accepted by siunitx."""
    if not math.isfinite(value):
        raise ValueError("Target summary contains a non-finite statistic")
    return rf"\num{{{value:.4e}}}"


def write_summary_tex(
    output_dir: Path,
    rows: list[dict[str, str | int | float]],
    matched_row_count: int,
) -> None:
    """Write a compact manuscript table of raw-space target quantiles."""
    lines = [
        r"% Requires \usepackage{booktabs,siunitx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Training-split distributions of the ten XRS targets in original target space. Statistics use finite target values among the {matched_row_count:,} training rows whose timestamps match the input Zarr; missing percentages use these matched rows as the denominator. Background-deviation targets retain negative values. Sample sums are unscaled sums of one-minute observations.}}",
        r"\label{tab:v1-target-distributions}",
        r"\begin{tabular}{l l r r r r r r r}",
        r"\toprule",
        r"Target & Scale & $N$ & Missing (\%) & $p_{25}$ & Median & $p_{75}$ & $p_{99}$ & Negative (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        target = TARGET_LABELS[str(row["target"])].replace(" · ", ", ")
        scale_label = (
            r"$\mathrm{W\,m^{-2}}$"
            if row["units"] == "W m^-2"
            else "sample sum"
        )
        row_tex = (
            f"{target} & {scale_label} & {row['n_train']} & "
            f"{float(row['missing_percent']):.2f} & {tex_num(float(row['p25']))} & "
            f"{tex_num(float(row['median']))} & {tex_num(float(row['p75']))} & "
            f"{tex_num(float(row['p99']))} & {float(row['negative_percent']):.2f} "
        )
        lines.append(row_tex + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    _ = (output_dir / "target_distribution_summary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_metadata(
    output_dir: Path,
    train_csv: Path,
    stats_path: Path,
    stats: StatsDocument,
    zarr_hashes: dict[str, str],
    histogram_bins: int,
) -> None:
    """Record exact input provenance and the plotting cohort."""
    train_csv_record = train_csv.resolve().relative_to(ROOT).as_posix()
    stats_record = stats_path.resolve().relative_to(ROOT).as_posix()
    metadata = {
        "fit_split": "train",
        "train_csv": train_csv_record,
        "train_csv_sha256": stats.source_csv_sha256,
        "stats_json": stats_record,
        "stats_json_sha256": sha256_file(stats_path),
        "zarr_time_sha256_by_year": zarr_hashes,
        "cohort_rule": stats.fit_cohort,
        "cohort_counts": {
            "train_rows": stats.train_rows,
            "input_timestamp_matches": stats.matched_rows,
            "missing_input_timestamp_rows": stats.train_rows - stats.matched_rows,
            "target_valid_rows": stats.target_valid_rows,
            "target_missing_rows": {
                name: stats.matched_rows - count
                for name, count in stats.target_valid_rows.items()
            },
        },
        "plots": [
            "10-panel empirical CDF of asinh(y / s), before per-target z-score normalization",
            "10-panel density histogram of asinh(y / s), before per-target z-score normalization",
        ],
        "histogram_bins": histogram_bins,
        "histogram_bin_edge_rule": "one shared set of equal-width bins spanning all transformed training values",
        "summary_table_space": "original target space; finite values only",
        "test_or_validation_data_used": False,
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Generate and save the V1 target-distribution artifacts."""
    args = parse_args()
    stats = load_stats(args.stats_json)
    if args.histogram_bins < 10:
        raise ValueError("At least ten histogram bins are required")
    rows, zarr_hashes = matched_rows(args.train_csv, args.zarr_path, stats)
    distributions, summaries = summarize_targets(rows, stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_ecdf_figure(args.output_dir, distributions)
    save_histogram_figure(args.output_dir, distributions, args.histogram_bins)
    write_summary_csv(args.output_dir, summaries)
    write_summary_tex(args.output_dir, summaries, stats.matched_rows)
    write_metadata(
        args.output_dir,
        args.train_csv,
        args.stats_json,
        stats,
        zarr_hashes,
        args.histogram_bins,
    )
    logger.info(
        "Saved V1 target distributions for {} targets from {} matched train rows to {}",
        len(TARGET_COLUMNS),
        len(rows),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
