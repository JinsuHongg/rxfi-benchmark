"""Compare legacy event RXFI with the XRS-only background deviation target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

SPLITS = ("train", "validation", "leaky_validation", "test")
NEW_TARGET = "max_background_deviation"
OLD_TARGET = "max_old_rxfi"
REQUIRED_NEW_COLUMNS = {"timestamp", "max_goes_class", NEW_TARGET}
REQUIRED_OLD_COLUMNS = {
    "timestamp",
    OLD_TARGET,
    "old_rxfi_target_status",
    "old_rxfi_event_count_24h",
    "valid_old_rxfi_event_count_24h",
}
FIGURE_NAME = "old_rxfi_vs_background_deviation"
CLASS_ORDER = ("FQ", "A", "B", "C", "M", "X")
CLASS_COLORS = {
    "FQ": "#777777",
    "A": "#56B4E9",
    "B": "#0072B2",
    "C": "#E69F00",
    "M": "#D55E00",
    "X": "#CC79A7",
}


@dataclass(frozen=True)
class Pair:
    """One aligned forecast window and its available label values."""

    timestamp: datetime
    deviation: float | None
    goes_class: str
    old_rxfi: float | None
    old_status: str
    event_count: int
    valid_event_count: int


def finite_float(value: str | None) -> float | None:
    """Parse a finite number while preserving missing values as None."""
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def signed_log_deviation(values: np.ndarray, scale: float) -> np.ndarray:
    """Transform signed flux deviations while preserving zero and sign.

    Args:
        values: Background deviations in W m^-2.
        scale: Positive reference deviation in W m^-2.

    Returns:
        Dimensionless signed-log transformed deviations.
    """
    return np.sign(values) * np.log10(1.0 + np.abs(values) / scale)


def parse_timestamp(value: str) -> datetime:
    """Parse a split timestamp and normalize it to UTC."""
    result = datetime.fromisoformat(value.strip())
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def parse_count(value: str | None) -> int:
    """Parse a nonnegative event count, using zero for an empty audit field."""
    if value is None or not value.strip():
        return 0
    return int(value)


def parse_goes_class(value: str | None) -> str:
    """Map a stored class label to its letter class while preserving FQ."""
    label = (value or "").strip().upper()
    if label == "FQ":
        return "FQ"
    category = label[:1]
    if category not in CLASS_ORDER:
        raise ValueError(f"Unsupported GOES class label: {value!r}")
    return category


def read_csv(path: Path, required: set[str]) -> dict[datetime, dict[str, str | None]]:
    """Read a CSV keyed by unique UTC timestamp and validate its schema."""
    rows: dict[datetime, dict[str, str | None]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            timestamp = parse_timestamp(row["timestamp"] or "")
            if timestamp in rows:
                raise ValueError(f"Duplicate timestamp in {path}: {timestamp.isoformat()}")
            rows[timestamp] = row
    return rows


def load_split(primary_path: Path, old_path: Path) -> list[Pair]:
    """Align one current target split with its supplementary old-RXFI copy."""
    primary_rows = read_csv(primary_path, REQUIRED_NEW_COLUMNS)
    old_rows = read_csv(old_path, REQUIRED_OLD_COLUMNS)
    if primary_rows.keys() != old_rows.keys():
        raise ValueError(f"Split timestamps do not align: {primary_path} and {old_path}")
    pairs: list[Pair] = []
    for timestamp, primary in primary_rows.items():
        old = old_rows[timestamp]
        pairs.append(
            Pair(
                timestamp=timestamp,
                deviation=finite_float(primary[NEW_TARGET]),
                goes_class=parse_goes_class(primary["max_goes_class"]),
                old_rxfi=finite_float(old[OLD_TARGET]),
                old_status=(old["old_rxfi_target_status"] or ""),
                event_count=parse_count(old["old_rxfi_event_count_24h"]),
                valid_event_count=parse_count(old["valid_old_rxfi_event_count_24h"]),
            )
        )
    return sorted(pairs, key=lambda pair: pair.timestamp)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    """Write a CSV with a stable column order."""
    records = list(rows)
    fieldnames = list(dict.fromkeys([*fieldnames, *(key for row in records for key in row)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def sha256_file(path: Path) -> str:
    """Hash a source file incrementally for analysis provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_counts(values: list[float], category: str) -> dict[str, int]:
    """Count values around the label's prespecified reference point."""
    if category == "ratio":
        return {
            "below_reference": sum(value < 1.0 for value in values),
            "at_reference": sum(value == 1.0 for value in values),
            "above_reference": sum(value > 1.0 for value in values),
        }
    return {
        "negative": sum(value < 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
        "positive": sum(value > 0.0 for value in values),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    """Compute descriptive statistics for one target on the paired cohort."""
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(np.max(array)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> float:
    """Compute Spearman correlation from average ranks."""
    rank_x, rank_y = average_ranks(values_x), average_ranks(values_y)
    if np.ptp(rank_x) == 0.0 or np.ptp(rank_y) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def block_bootstrap_ci(
    pairs: list[tuple[Pair, float, float]],
    block_days: int,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    """Estimate a percentile interval by resampling contiguous date blocks."""
    if replicates <= 0:
        return float("nan"), float("nan"), 0
    if not pairs:
        return float("nan"), float("nan"), 0
    origin: date = pairs[0][0].timestamp.date()
    grouped: dict[int, list[tuple[Pair, float, float]]] = defaultdict(list)
    for row in pairs:
        pair = row[0]
        offset = (pair.timestamp.date() - origin).days
        grouped[offset // block_days].append(row)
    blocks = list(grouped.values())
    rng = random.Random(seed)
    correlations: list[float] = []
    for _ in range(replicates):
        sample: list[tuple[Pair, float, float]] = []
        for block_index in (rng.randrange(len(blocks)) for _ in blocks):
            sample.extend(blocks[block_index])
        sample = sample[: len(pairs)]
        corr = spearman(
            np.asarray([row[2] for row in sample], dtype=float),
            np.asarray([row[1] for row in sample], dtype=float),
        )
        if math.isfinite(corr):
            correlations.append(corr)
    if not correlations:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(np.asarray(correlations), [0.025, 0.975])
    return float(low), float(high), len(correlations)


def split_coverage(split: str, pairs: list[Pair]) -> dict[str, object]:
    """Summarize data availability and old-catalog status for one split."""
    both = [pair for pair in pairs if pair.deviation is not None and pair.old_rxfi is not None]
    statuses = Counter(pair.old_status for pair in pairs)
    return {
        "split": split,
        "windows": len(pairs),
        "valid_max_background_deviation": sum(pair.deviation is not None for pair in pairs),
        "valid_max_old_rxfi": sum(pair.old_rxfi is not None for pair in pairs),
        "paired_valid": len(both),
        "deviation_only": sum(pair.deviation is not None and pair.old_rxfi is None for pair in pairs),
        "old_rxfi_only": sum(pair.deviation is None and pair.old_rxfi is not None for pair in pairs),
        "both_missing": sum(pair.deviation is None and pair.old_rxfi is None for pair in pairs),
        "no_catalog_flare": statuses["no_catalog_flare"],
        "no_valid_old_rxfi": statuses["no_valid_old_rxfi"],
        "old_rxfi_ok": statuses["ok"],
    }


def paired_rows(pairs: list[Pair]) -> list[tuple[Pair, float, float]]:
    """Return rows with both values present and narrowed to finite floats."""
    result: list[tuple[Pair, float, float]] = []
    for pair in pairs:
        if pair.deviation is None or pair.old_rxfi is None:
            continue
        result.append((pair, pair.deviation, pair.old_rxfi))
    return result


def paired_summaries(split: str, pairs: list[Pair]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build descriptive target summaries and the reference-category table."""
    cohort = paired_rows(pairs)
    deviation = [row[1] for row in cohort]
    old_rxfi = [row[2] for row in cohort]
    summary_rows: list[dict[str, object]] = []
    for target, values, category in (
        (NEW_TARGET, deviation, "signed"),
        (OLD_TARGET, old_rxfi, "ratio"),
    ):
        row: dict[str, object] = {"split": split, "target": target, **summarize(values)}
        row.update(category_counts(values, category))
        summary_rows.append(row)
    cells = Counter(
        (
            "below_1" if old_value < 1.0 else "at_1" if old_value == 1.0 else "above_1",
            "negative" if deviation_value < 0.0 else "zero" if deviation_value == 0.0 else "positive",
        )
        for _, deviation_value, old_value in cohort
    )
    cross_rows: list[dict[str, object]] = [
        {
            "split": split,
            "old_rxfi_category": old_category,
            "background_deviation_category": deviation_category,
            "count": cells[(old_category, deviation_category)],
            "paired_valid": len(cohort),
        }
        for old_category in ("below_1", "at_1", "above_1")
        for deviation_category in ("negative", "zero", "positive")
    ]
    return summary_rows, cross_rows


def association_row(
    split: str,
    pairs: list[Pair],
    block_days: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Summarize paired rank association and block-bootstrap uncertainty."""
    observed = paired_rows(pairs)
    x = np.asarray([row[2] for row in observed], dtype=float)
    y = np.asarray([row[1] for row in observed], dtype=float)
    rho = spearman(x, y) if len(observed) > 1 else float("nan")
    low, high, valid_replicates = block_bootstrap_ci(observed, block_days, replicates, seed)
    return {
        "split": split,
        "paired_valid": len(observed),
        "spearman_rho": rho,
        "block_days": block_days,
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": valid_replicates,
        "ci_95_low": low,
        "ci_95_high": high,
    }


def plot_comparison(
    data: dict[str, list[Pair]], output_dir: Path, deviation_scale: float
) -> list[str]:
    """Save split-wise paired distribution and relationship figures."""
    figure, axes = plt.subplots(3, 4, figsize=(17, 11))
    figure.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.84, hspace=0.43, wspace=0.34)
    hexagons: list[PolyCollection] = []
    deviation_axis_label = "Signed log deviation"
    for column, split in enumerate(SPLITS):
        pairs = paired_rows(data[split])
        valid_deviation_rows = [pair for pair in data[split] if pair.deviation is not None]
        valid_old_rxfi_rows = [
            pair for pair in data[split] if pair.old_rxfi is not None and pair.old_rxfi > 0.0
        ]
        old_values = np.asarray([row[2] for row in pairs], dtype=float)
        deviation_values = np.asarray([row[1] for row in pairs], dtype=float)
        deviation_axis = axes[0, column]
        rxfi_axis = axes[1, column]
        relation_axis = axes[2, column]
        transformed_deviation = signed_log_deviation(deviation_values, deviation_scale)
        class_deviations = [
            signed_log_deviation(
                np.asarray(
                    [
                        pair.deviation
                        for pair in valid_deviation_rows
                        if pair.goes_class == category
                    ],
                    dtype=float,
                ),
                deviation_scale,
            )
            for category in CLASS_ORDER
        ]
        deviation_axis.hist(
            class_deviations,
            bins=50,
            color=[CLASS_COLORS[category] for category in CLASS_ORDER],
            alpha=0.65,
            stacked=True,
        )
        deviation_axis.axvline(0.0, color="#222222", linewidth=1, linestyle="--")
        deviation_axis.set_title(f"{split} (valid n={len(valid_deviation_rows):,})")
        deviation_axis.set_xlabel(deviation_axis_label)
        deviation_axis.set_ylabel("Window count")

        class_rxfi_values = [
            np.asarray(
                [
                    pair.old_rxfi
                    for pair in valid_old_rxfi_rows
                    if pair.goes_class == category
                ],
                dtype=float,
            )
            for category in CLASS_ORDER
        ]
        positive_old = np.concatenate(class_rxfi_values)
        if positive_old.size:
            lower = float(np.min(positive_old))
            upper = float(np.max(positive_old))
            bins = np.geomspace(lower * 0.999, upper * 1.001, 50) if upper > lower else 50
            rxfi_axis.hist(
                class_rxfi_values,
                bins=bins,
                color=[CLASS_COLORS[category] for category in CLASS_ORDER],
                alpha=0.65,
                stacked=True,
            )
            rxfi_axis.set_xscale("log")
        rxfi_axis.axvline(1.0, color="#222222", linewidth=1, linestyle="--")
        rxfi_axis.set_xlabel("Max old RXFI (dimensionless)")
        rxfi_axis.set_ylabel("Window count")
        rxfi_axis.set_title(f"Legacy RXFI (valid n={len(valid_old_rxfi_rows):,})")

        if pairs:
            hexagon = relation_axis.hexbin(
                np.log10(np.maximum(old_values, np.finfo(float).tiny)),
                transformed_deviation,
                gridsize=45,
                mincnt=1,
                cmap="viridis",
                norm=LogNorm(vmin=1, vmax=2),
            )
            hexagons.append(hexagon)
            relation_axis.axvline(0.0, color="#222222", linewidth=1, linestyle="--")
            relation_axis.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
        relation_axis.set_xlabel("log$_{10}$(max old RXFI)")
        relation_axis.set_ylabel(deviation_axis_label)
        relation_axis.set_title("Paired windows")
    if hexagons:
        bin_counts: list[float] = []
        for hexagon in hexagons:
            counts = hexagon.get_array()
            if counts is not None and counts.size:
                bin_counts.append(float(np.max(counts)))
        highest_bin_count = max(bin_counts, default=1.0)
        shared_norm = LogNorm(vmin=1, vmax=max(2.0, highest_bin_count))
        for hexagon in hexagons:
            hexagon.set_norm(shared_norm)
        figure.colorbar(  # pyright: ignore[reportUnknownMemberType]
            hexagons[-1],
            ax=axes[2, :].tolist(),
            label="Paired windows per hexagon",
        )
    figure.suptitle(  # pyright: ignore[reportUnknownMemberType]
        "Legacy RXFI and XRS-only background deviation\n"
        "sgn(D) log10(1 + |D| / D0), "
        f"D0={deviation_scale:.1e} W m^-2",
        fontsize=15,
        y=0.99,
    )
    figure.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=[Patch(facecolor=CLASS_COLORS[label], label=label) for label in CLASS_ORDER],
        title="Max GOES class",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(CLASS_ORDER),
        frameon=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for extension in ("svg", "png"):
        path = output_dir / f"{FIGURE_NAME}.{extension}"
        figure.savefig(  # pyright: ignore[reportUnknownMemberType]
            path, dpi=220 if extension == "png" else None, bbox_inches="tight"
        )
        saved.append(str(path))
    plt.close(figure)
    return saved


def run(args: argparse.Namespace) -> None:
    """Run the paired-label comparison and persist tables, figures, and provenance."""
    if args.block_days < 1 or args.bootstrap_replicates < 0:
        raise ValueError("Block length must be positive and bootstrap repetitions nonnegative")
    if args.deviation_scale <= 0.0:
        raise ValueError("Deviation scale must be positive")
    data = {
        split: load_split(
            args.input_dir / f"{split}_targets.csv",
            args.old_rxfi_dir / f"{split}_targets.csv",
        )
        for split in SPLITS
    }
    coverage_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    input_paths: list[Path] = []
    for split in SPLITS:
        input_paths.extend(
            [
                args.input_dir / f"{split}_targets.csv",
                args.old_rxfi_dir / f"{split}_targets.csv",
            ]
        )
        pairs = data[split]
        coverage_rows.append(split_coverage(split, pairs))
        split_summaries, split_cross = paired_summaries(split, pairs)
        summary_rows.extend(split_summaries)
        cross_rows.extend(split_cross)
        association_rows.append(
            association_row(split, pairs, args.block_days, args.bootstrap_replicates, args.seed)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_coverage.csv", list(coverage_rows[0]), coverage_rows)
    write_csv(args.output_dir / "paired_target_summaries.csv", list(summary_rows[0]), summary_rows)
    write_csv(args.output_dir / "paired_reference_cross_tab.csv", list(cross_rows[0]), cross_rows)
    write_csv(args.output_dir / "paired_association.csv", list(association_rows[0]), association_rows)
    figures = plot_comparison(data, args.output_dir / "figures", args.deviation_scale)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    documentation_figures: list[str] = []
    for figure in figures:
        source = Path(figure)
        destination = args.figure_dir / source.name
        shutil.copy2(source, destination)
        documentation_figures.append(str(destination))
    metadata = {
        "analysis": "exploratory paired comparison of legacy old RXFI and XRS-only max background deviation",
        "old_rxfi_target": "maximum finite catalog old_rxfi among events whose start_time is in [t,t+24h)",
        "background_deviation_target": "max_i(F_i-B_s), with B_s the lower-50% mean over [t-24h,t)",
        "paired_cohort": "both target values are finite; missing labels are never imputed",
        "deviation_histogram_cohort": "all rows with finite max_background_deviation; stacked by max_goes_class letter, preserving FQ",
        "old_rxfi_histogram_cohort": "all rows with positive finite max_old_rxfi; stacked by max_goes_class letter, preserving FQ",
        "reference_categories": {"old_rxfi": "<1, =1, >1", "background_deviation": "<0, =0, >0"},
        "uncertainty": "95% percentile interval from contiguous date-block bootstrap; descriptive, not a significance test",
        "block_days": args.block_days,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.seed,
        "deviation_plot_transform": "sign(D) * log10(1 + abs(D) / D0); D is max_background_deviation in W m^-2",
        "deviation_plot_scale_D0_w_m2": args.deviation_scale,
        "histogram_count_axis": "linear; deviation and old RXFI distributions stacked by max_goes_class",
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "figures": figures,
        "documentation_figures": documentation_figures,
        "note": "The test split was inspected descriptively in this audit; any target choice informed by these results is no longer independent of the existing test cohort and needs a future independent evaluation.",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved paired-label comparison to {}", args.output_dir)


def main() -> None:
    """Parse options and run the reproducible comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--old-rxfi-dir", type=Path, default=Path("data/derived/old_rxfi_targets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/old_rxfi_background_deviation_comparison"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/research/assets"))
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--deviation-scale", type=float, default=1e-6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
