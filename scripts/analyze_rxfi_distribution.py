#!/usr/bin/env python3
"""Validate immutable Surya splits and describe NOAA-derived RXFI distributions.

The split CSVs are read only.  NOAA events are matched as a documented,
deterministic *candidate* association: for each non-quiet sample, select the
highest NOAA XRS-B peak in [timestamp, timestamp + 24 hours).  The 24-hour
look-ahead is an explicit analysis assumption, not an alteration of a split.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from inspect_noaa_xrs_flare_report import calculate_rxfi


DEFAULT_DATA_DIR = Path("/mnt/storage/surya/index_data")
DEFAULT_NOAA_CSV = Path("/tmp/rxfi-noaa/sci_xrsf-l2-flrpt_geo_s19950103_e20260913_v1-0-1.csv")
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
    "leaky_validation": "leaky_validation.csv",
}
SPLIT_RULES = {
    "train": "2010-2019, February 15 through December 31",
    "validation": "2010-2019, January 15 through January 31",
    "test": "2020-2024, all dates",
    "leaky_validation": "2010-2019, January 1-14 and February 1-14",
}
CLASS_ORDER = ("FQ", "A", "B", "C", "M", "X")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def percentile(values: list[float], probability: float) -> float:
    """Linearly interpolated quantile for a sorted nonempty list."""
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def class_group(label: str) -> str:
    """Map a split label to a conventional GOES letter or the observed FQ code."""
    value = label.strip().upper()
    if value == "FQ":
        return "FQ"
    return value[:1] if value[:1] in CLASS_ORDER else "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samples(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def read_noaa_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            peak_time = datetime.fromisoformat(row["time"])
            if 2010 <= peak_time.year <= 2024:
                try:
                    peak = float(row["xrsb_irrad"])
                except (TypeError, ValueError):
                    continue
                events.append(
                    {
                        "time": peak_time,
                        "peak": peak,
                        "class": class_group(row.get("flare_class", "")),
                        "rxfi": calculate_rxfi(row.get("xrsb_irrad"), row.get("background_irrad")),
                    }
                )
    return events


def split_date_valid(split: str, timestamp: datetime) -> bool:
    if split == "train":
        return 2010 <= timestamp.year <= 2019 and (timestamp.month, timestamp.day) >= (2, 15)
    if split == "validation":
        return 2010 <= timestamp.year <= 2019 and timestamp.month == 1 and timestamp.day >= 15
    if split == "test":
        return 2020 <= timestamp.year <= 2024
    return 2010 <= timestamp.year <= 2019 and (
        (timestamp.month == 1) or (timestamp.month == 2 and timestamp.day <= 14)
    )


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "median": None,
                **{f"p{int(q * 100):02d}": None for q in QUANTILES}}
    result: dict[str, float | int | None] = {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
        "std": statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        "median": statistics.median(sorted_values),
    }
    result.update({f"p{int(q * 100):02d}": percentile(sorted_values, q) for q in QUANTILES})
    return result


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        rank = (position + end + 2) / 2.0
        for offset in range(position, end + 1):
            ranks[order[offset]] = rank
        position = end + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#18212b}.small{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:bold}.axis{stroke:#425466;stroke-width:1}.grid{stroke:#d9e1e8;stroke-width:1}.bar{fill:#2b6cb0}.train{fill:#2b6cb0}.validation{fill:#38a169}.test{fill:#d69e2e}</style>',
        f'<text x="40" y="28" class="title">{title}</text>',
    ]


def save_svg(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def class_distribution_svg(path: Path, distributions: dict[str, Counter[str]]) -> None:
    width, height, left, bottom = 900, 410, 80, 70
    chart_w, chart_h = width - left - 40, height - bottom - 55
    parts = svg_start(width, height, "Conventional flare-class distribution by frozen split")
    groups = ["train", "validation", "test"]
    for tick in range(0, 101, 20):
        y = height - bottom - chart_h * tick / 100
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick}%</text>')
    for index, flare_class in enumerate(CLASS_ORDER):
        cx = left + chart_w * (index + 0.5) / len(CLASS_ORDER)
        parts.append(f'<text x="{cx:.1f}" y="{height - 42}" text-anchor="middle" class="label">{flare_class}</text>')
        for group_index, split in enumerate(groups):
            total = sum(distributions[split].values())
            percent = 100 * distributions[split][flare_class] / total if total else 0
            bar_w = chart_w / len(CLASS_ORDER) / 4
            x = cx + (group_index - 1) * bar_w - bar_w / 2
            bar_h = chart_h * percent / 100
            y = height - bottom - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" class="{split}"/>')
    parts.extend([f'<line x1="{left}" y1="{height-bottom}" x2="{left+chart_w}" y2="{height-bottom}" class="axis"/>',
                  f'<line x1="{left}" y1="{height-bottom-chart_h}" x2="{left}" y2="{height-bottom}" class="axis"/>',
                  '<text x="450" y="398" text-anchor="middle" class="small">FQ denotes the observed quiet/no-class label in the supplied split files.</text>'])
    for index, split in enumerate(groups):
        x = 590 + index * 95
        parts.append(f'<rect x="{x}" y="38" width="12" height="12" class="{split}"/><text x="{x+17}" y="49" class="small">{split}</text>')
    save_svg(path, parts)


def histogram_svg(path: Path, values: list[float]) -> None:
    width, height, left, bottom = 920, 410, 70, 55
    parts = svg_start(width, height, "RXFI distribution: raw (left) and log10(1 + RXFI) (right)")
    panels = [(90, "Raw RXFI; display range ends at p99"), (510, "log10(1 + RXFI); full range")]
    for panel, (x0, label) in enumerate(panels):
        chart_w, chart_h = 320, 255
        source = values if panel == 0 else [math.log10(1 + value) for value in values]
        upper = percentile(source, 0.99) if panel == 0 else max(source)
        bins = [0] * 30
        for value in source:
            bucket = min(29, int(30 * value / upper)) if upper > 0 else 0
            bins[bucket] += 1
        max_count = max(bins) or 1
        for i, count in enumerate(bins):
            bar_w = chart_w / len(bins)
            h = chart_h * count / max_count
            parts.append(f'<rect x="{x0+i*bar_w:.1f}" y="{bottom+chart_h-h:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" class="bar"/>')
        parts.extend([f'<line x1="{x0}" y1="{bottom+chart_h}" x2="{x0+chart_w}" y2="{bottom+chart_h}" class="axis"/>',
                      f'<line x1="{x0}" y1="{bottom}" x2="{x0}" y2="{bottom+chart_h}" class="axis"/>',
                      f'<text x="{x0+chart_w/2:.1f}" y="{bottom+chart_h+24}" text-anchor="middle" class="small">{label}</text>',
                      f'<text x="{x0}" y="{bottom+chart_h+42}" class="small">0</text>',
                      f'<text x="{x0+chart_w}" y="{bottom+chart_h+42}" text-anchor="end" class="small">{upper:.3g}</text>'])
    parts.append('<text x="460" y="390" text-anchor="middle" class="small">Raw values above p99 are included in all summaries but assigned to the final display bin in the left panel.</text>')
    save_svg(path, parts)


def class_boxplot_svg(path: Path, values_by_class: dict[str, list[float]]) -> None:
    classes = [item for item in CLASS_ORDER if item != "FQ" and values_by_class[item]]
    width, height, left, bottom = 820, 430, 85, 70
    chart_w, chart_h = width - left - 45, height - bottom - 50
    transformed = {key: sorted(math.log10(1 + value) for value in values_by_class[key]) for key in classes}
    ymax = max(percentile(values, 0.90) for values in transformed.values())
    parts = svg_start(width, height, "RXFI conditional on NOAA conventional flare class")
    for tick in range(0, 6):
        val = ymax * tick / 5
        y = height - bottom - chart_h * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" class="grid"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{val:.2f}</text>')
    for index, flare_class in enumerate(classes):
        values = transformed[flare_class]
        q10, q25, q50, q75, q90 = (percentile(values, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
        x = left + chart_w * (index + 0.5) / len(classes)
        scale = lambda value: height - bottom - chart_h * value / ymax
        parts.extend([f'<line x1="{x}" y1="{scale(q10):.1f}" x2="{x}" y2="{scale(q90):.1f}" class="axis"/>',
                      f'<line x1="{x-16}" y1="{scale(q10):.1f}" x2="{x+16}" y2="{scale(q10):.1f}" class="axis"/>',
                      f'<line x1="{x-16}" y1="{scale(q90):.1f}" x2="{x+16}" y2="{scale(q90):.1f}" class="axis"/>',
                      f'<rect x="{x-22}" y="{scale(q75):.1f}" width="44" height="{scale(q25)-scale(q75):.1f}" fill="#90cdf4" stroke="#2b6cb0"/>',
                      f'<line x1="{x-22}" y1="{scale(q50):.1f}" x2="{x+22}" y2="{scale(q50):.1f}" stroke="#1a365d" stroke-width="2"/>',
                      f'<text x="{x}" y="{height-42}" text-anchor="middle" class="label">{flare_class}</text>'])
    parts.extend([f'<line x1="{left}" y1="{height-bottom}" x2="{left+chart_w}" y2="{height-bottom}" class="axis"/>',
                  f'<line x1="{left}" y1="{height-bottom-chart_h}" x2="{left}" y2="{height-bottom}" class="axis"/>',
                  '<text x="410" y="404" text-anchor="middle" class="small">Vertical scale: log10(1 + RXFI); box = Q1–Q3, line = median, whiskers = p10–p90.</text>'])
    save_svg(path, parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--noaa-csv", type=Path, default=DEFAULT_NOAA_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rxfi_distribution"))
    args = parser.parse_args()
    if not args.noaa_csv.is_file():
        raise SystemExit(f"NOAA CSV not found: {args.noaa_csv}")

    output_dir = args.output_dir
    tables_dir, figures_dir = output_dir / "tables", output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    events = read_noaa_events(args.noaa_csv)
    event_times = [event["time"] for event in events]
    split_rows: dict[str, list[dict[str, str]]] = {}
    split_classes: dict[str, Counter[str]] = {}
    split_rxfi: dict[str, list[float]] = defaultdict(list)
    class_rxfi: dict[str, list[float]] = defaultdict(list)
    split_summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    all_match_class_codes: list[float] = []
    all_match_log_rxfi: list[float] = []

    for split, filename in SPLIT_FILES.items():
        path = args.data_dir / filename
        if not path.is_file():
            raise SystemExit(f"Split not found: {path}")
        rows, columns = read_samples(path)
        required = {"timestamp", "max_goes_class", "max_intensity"}
        missing_columns = sorted(required.difference(columns))
        timestamps = [datetime.fromisoformat(row["timestamp"]) for row in rows if row.get("timestamp")]
        missing_key = {key: sum(not row.get(key, "").strip() for row in rows) for key in required}
        duplicate_timestamps = len(timestamps) - len(set(timestamps))
        invalid_dates = sum(not split_date_valid(split, timestamp) for timestamp in timestamps)
        classes = Counter(class_group(row.get("max_goes_class", "")) for row in rows)
        split_rows[split], split_classes[split] = rows, classes
        split_summary_rows.append({
            "split": split, "file": str(path), "sha256_read_only": sha256(path), "rows": len(rows),
            "min_timestamp": min(timestamps).isoformat(sep=" ") if timestamps else "", "max_timestamp": max(timestamps).isoformat(sep=" ") if timestamps else "",
            "duplicate_timestamps": duplicate_timestamps, "rows_outside_frozen_rule": invalid_dates,
            "missing_required_columns": ";".join(missing_columns), **{f"missing_{key}": value for key, value in missing_key.items()},
            "counts_by_year": json.dumps(dict(sorted(Counter(item.year for item in timestamps).items()))),
        })
        for flare_class in list(CLASS_ORDER) + (["other"] if classes["other"] else []):
            class_rows.append({"split": split, "class": flare_class, "count": classes[flare_class], "percent": 100 * classes[flare_class] / len(rows) if rows else 0})

        candidates = unique = ambiguous = no_candidate = quiet = invalid_rxfi = class_disagreement = 0
        for row in rows:
            label = class_group(row.get("max_goes_class", ""))
            if label == "FQ":
                quiet += 1
                continue
            timestamp = datetime.fromisoformat(row["timestamp"])
            lo = bisect.bisect_left(event_times, timestamp)
            hi = bisect.bisect_left(event_times, timestamp + timedelta(hours=24))
            window = events[lo:hi]
            if not window:
                no_candidate += 1
                continue
            candidates += 1
            largest_peak = max(event["peak"] for event in window)
            maxima = [event for event in window if event["peak"] == largest_peak]
            if len(maxima) != 1:
                ambiguous += 1
                continue
            event = maxima[0]
            unique += 1
            if event["class"] != label:
                class_disagreement += 1
            if event["rxfi"] is None:
                invalid_rxfi += 1
                continue
            rxfi = float(event["rxfi"])
            split_rxfi[split].append(rxfi)
            class_rxfi[event["class"]].append(rxfi)
            if event["class"] in {"A", "B", "C", "M", "X"}:
                all_match_class_codes.append(float(("A", "B", "C", "M", "X").index(event["class"]) + 1))
                all_match_log_rxfi.append(math.log10(1 + rxfi))
        matching_rows.append({"split": split, "rows": len(rows), "quiet_no_flare_target": quiet, "candidate_window_nonquiet": candidates,
                              "unique_candidate_matches": unique, "ambiguous_maxima": ambiguous, "no_noaa_event_in_window": no_candidate,
                              "invalid_rxfi_after_match": invalid_rxfi, "valid_rxfi": len(split_rxfi[split]),
                              "valid_rxfi_percent_all_rows": 100 * len(split_rxfi[split]) / len(rows) if rows else 0,
                              "matched_class_disagreement": class_disagreement})

    rxfi_split_rows: list[dict[str, Any]] = []
    for split in SPLIT_FILES:
        stats = describe(split_rxfi[split])
        stats["split"] = split
        stats["missing_or_unmatched"] = len(split_rows[split]) - len(split_rxfi[split])
        rxfi_split_rows.append(stats)
    rxfi_class_rows: list[dict[str, Any]] = []
    for flare_class in ("A", "B", "C", "M", "X"):
        raw = describe(class_rxfi[flare_class])
        logged = describe([math.log10(1 + value) for value in class_rxfi[flare_class]])
        rxfi_class_rows.append({"class": flare_class, **{f"raw_{key}": value for key, value in raw.items()}, **{f"log10_1p_{key}": value for key, value in logged.items()}})

    correlation = spearman(all_match_class_codes, all_match_log_rxfi)
    write_csv(tables_dir / "split_validation_summary.csv", split_summary_rows)
    write_csv(tables_dir / "flare_class_distribution.csv", class_rows)
    write_csv(tables_dir / "matching_summary.csv", matching_rows)
    write_csv(tables_dir / "rxfi_summary_by_split.csv", rxfi_split_rows)
    write_csv(tables_dir / "rxfi_summary_by_class.csv", rxfi_class_rows)
    analysis_summary = {
        "noaa_csv": str(args.noaa_csv), "noaa_event_count_2010_2024": len(events), "matching_assumption": "highest NOAA XRS-B peak in [sample timestamp, timestamp + 24 hours), excluding FQ; tied maxima are ambiguous",
        "spearman_ordered_noaa_class_vs_log10_1p_rxfi": correlation, "split_rules": SPLIT_RULES,
    }
    (tables_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2) + "\n", encoding="utf-8")
    all_valid = [value for values in split_rxfi.values() for value in values]
    class_distribution_svg(figures_dir / "figure_class_distribution.svg", split_classes)
    histogram_svg(figures_dir / "figure_rxfi_distribution.svg", all_valid)
    class_boxplot_svg(figures_dir / "figure_rxfi_by_flare_class.svg", class_rxfi)
    print(json.dumps({"split_rows": {key: len(value) for key, value in split_rows.items()}, "valid_rxfi": {key: len(value) for key, value in split_rxfi.items()}, "spearman": correlation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
