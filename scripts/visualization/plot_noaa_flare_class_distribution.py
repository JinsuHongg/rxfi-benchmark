#!/usr/bin/env python3
"""Plot event-level GOES flare-class counts from a NOAA flare-summary CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

try:
    from loguru import logger  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Liberation Sans"


CLASS_ORDER: Final[tuple[str, ...]] = ("A", "B", "C", "M", "X")
COLORS: Final[dict[str, str]] = {
    "A": "#c6dbef",
    "B": "#9ecae1",
    "C": "#6baed6",
    "M": "#3182bd",
    "X": "#08519c",
}


def sha256(path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_group(label: str) -> str:
    """Extract and validate the primary GOES flare class letter.

    Args:
        label: Raw flare class label string.

    Returns:
        Single-letter class identifier (A, B, C, M, or X).

    Raises:
        ValueError: If the label does not start with a recognized class.
    """
    value = label.strip().upper()
    if value[:1] not in CLASS_ORDER:
        raise ValueError(f"Unsupported flare_class: {label!r}")
    return value[:1]


def read_counts(path: Path, start: datetime, end: datetime) -> Counter[str]:
    """Read NOAA flare catalog and aggregate class occurrences within a time window.

    Args:
        path: Path to the NOAA flare-summary CSV file.
        start: Inclusive start timestamp.
        end: Exclusive end timestamp.

    Returns:
        Counter mapping flare classes to event counts.

    Raises:
        ValueError: If required columns ('time', 'flare_class') are missing.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "flare_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")
        counts: Counter[str] = Counter()
        for row in reader:
            peak = datetime.fromisoformat(row["time"])
            if start <= peak < end:
                counts[class_group(row["flare_class"])] += 1
    return counts


def write_figure(output_dir: Path, counts: Counter[str]) -> None:
    """Write presentation-ready SVG and PNG bar charts for event counts.

    Args:
        output_dir: Directory where SVG and PNG files will be written.
        counts: Counter mapping flare classes to event counts.
    """
    labels = list(CLASS_ORDER)
    values = [counts[flare_class] for flare_class in labels]
    total = sum(values)

    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    bars = axis.bar(
        labels,
        values,
        color=[COLORS[flare_class] for flare_class in labels],
        width=0.56,
    )
    axis.set_ylim(0, max(values) * 1.15)
    axis.set_xlabel("Flare Class", fontsize=18, fontweight="bold", labelpad=10)
    axis.set_ylabel("Event Count", fontsize=18, fontweight="bold", labelpad=10)
    axis.tick_params(axis="both", which="major", labelsize=14, width=1.2, length=5)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", color="#d9e1e8", linewidth=1.0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.bar_label(
        bars,
        labels=[f"{count:,} ({100 * count / total:.2f}%)" for count in values],
        padding=4,
        fontsize=14,
        fontweight="bold",
    )
    for extension in ("svg", "png"):
        figure.savefig(
            output_dir
            / f"figure_noaa_flare_summary_class_distribution_2010_2024.{extension}",
            dpi=300,
        )
    plt.close(figure)


def main() -> int:
    """Parse command-line arguments and execute the flare class analysis pipeline.

    Returns:
        Exit code indicating execution status (0 for success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/noaa_flare_summary_class_distribution"),
    )
    args = parser.parse_args()
    start, end = datetime(2010, 1, 1), datetime(2025, 1, 1)
    counts = read_counts(args.catalog_csv, start, end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "flare_class_distribution_2010_2024.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("class", "count", "percent"), lineterminator="\n"
        )
        writer.writeheader()
        total = sum(counts.values())
        for flare_class in CLASS_ORDER:
            writer.writerow(
                {
                    "class": flare_class,
                    "count": counts[flare_class],
                    "percent": f"{100 * counts[flare_class] / total:.8f}",
                }
            )
    metadata = {
        "catalog_csv": str(args.catalog_csv),
        "catalog_sha256": sha256(args.catalog_csv),
        "peak_time_filter": "2010-01-01T00:00:00 <= time < 2025-01-01T00:00:00",
        "class_definition": "first letter of NOAA flare_class",
        "event_count": sum(counts.values()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_figure(args.output_dir, counts)
    logger.info(f"Class distribution: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
