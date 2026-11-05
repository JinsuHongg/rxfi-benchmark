#!/usr/bin/env python3
"""Plot NOAA-derived GOES target-class distributions for all frozen splits.

The plot reads the confirmed ``max_flare_class`` target in each derived split,
not the historical ``max_goes_class`` field supplied with the source index.
FQ is retained as its own observed quiet/no-flare class.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


CLASS_ORDER = ("FQ", "A", "B", "C", "M", "X")
SPLITS = ("train", "validation", "test")
COLORS = {"FQ": "#718096", "A": "#90cdf4", "B": "#63b3ed", "C": "#4299e1", "M": "#2b6cb0", "X": "#1a365d"}


def class_group(label: str) -> str:
    """Return FQ or the conventional first-letter GOES class."""
    value = label.strip().upper()
    if value == "FQ":
        return "FQ"
    if value[:1] in CLASS_ORDER:
        return value[:1]
    raise ValueError(f"Unsupported max_flare_class value: {label!r}")


def read_distribution(path: Path) -> Counter[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "max_flare_class" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no max_flare_class column")
        return Counter(class_group(row["max_flare_class"]) for row in reader)


def write_csv(path: Path, distributions: dict[str, Counter[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "class", "count", "percent"), lineterminator="\n")
        writer.writeheader()
        for split in SPLITS:
            total = sum(distributions[split].values())
            for flare_class in CLASS_ORDER:
                count = distributions[split][flare_class]
                writer.writerow({"split": split, "class": flare_class, "count": count, "percent": f"{100 * count / total:.6f}"})


def svg_lines(distributions: dict[str, Counter[str]]) -> Iterable[str]:
    width, height = 1120, 600
    left, right, top, bottom = 116, 38, 75, 55
    panel_h, gap = 122, 47
    chart_w = width - left - right
    max_percent = max(100 * values[flare_class] / sum(values.values()) for values in distributions.values() for flare_class in CLASS_ORDER)
    ymax = max(10, int(max_percent / 10 + 1) * 10)
    yield '<?xml version="1.0" encoding="UTF-8"?>'
    yield f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    yield '<style>text{font-family:Arial,sans-serif;fill:#18212b}.title{font-size:19px;font-weight:bold}.subtitle{font-size:12px;fill:#4a5568}.split{font-size:15px;font-weight:bold}.small{font-size:11px}.axis{stroke:#425466;stroke-width:1}.grid{stroke:#d9e1e8;stroke-width:1}</style>'
    yield '<text x="40" y="28" class="title">NOAA-derived GOES target-class distribution by frozen split</text>'
    yield '<text x="40" y="47" class="subtitle">Target: max_flare_class; bars show percent within each split. Labels show count (percent). FQ = quiet/no flare.</text>'
    for panel, split in enumerate(SPLITS):
        y0 = top + panel * (panel_h + gap)
        total = sum(distributions[split].values())
        yield f'<text x="40" y="{y0 + 18}" class="split">{escape(split)} (n={total:,})</text>'
        for tick in range(0, ymax + 1, 10):
            y = y0 + panel_h - panel_h * tick / ymax
            yield f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>'
            yield f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick}%</text>'
        for index, flare_class in enumerate(CLASS_ORDER):
            slot = chart_w / len(CLASS_ORDER)
            bar_w = slot * 0.58
            x = left + index * slot + (slot - bar_w) / 2
            count = distributions[split][flare_class]
            percent = 100 * count / total
            bar_h = panel_h * percent / ymax
            y = y0 + panel_h - bar_h
            yield f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{COLORS[flare_class]}"/>'
            label_y = max(y - 6, y0 + 12)
            yield f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" class="small">{count:,} ({percent:.1f}%)</text>'
            if panel == len(SPLITS) - 1:
                yield f'<text x="{x + bar_w / 2:.1f}" y="{y0 + panel_h + 20}" text-anchor="middle" class="split">{flare_class}</text>'
        yield f'<line x1="{left}" y1="{y0 + panel_h}" x2="{left + chart_w}" y2="{y0 + panel_h}" class="axis"/>'
        yield f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0 + panel_h}" class="axis"/>'
    yield '<text x="560" y="586" text-anchor="middle" class="subtitle">Source: data/derived/{train,validation,test}_targets.csv; regenerated NOAA dataset contract.</text>'
    yield '</svg>'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/target_class_distribution"))
    args = parser.parse_args()
    distributions = {split: read_distribution(args.derived_dir / f"{split}_targets.csv") for split in SPLITS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "class_distribution_all_splits.csv", distributions)
    (args.output_dir / "figure_class_distribution_all_splits.svg").write_text("\n".join(svg_lines(distributions)) + "\n", encoding="utf-8")
    print({split: dict(distributions[split]) for split in SPLITS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
