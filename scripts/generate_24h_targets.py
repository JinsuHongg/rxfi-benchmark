#!/usr/bin/env python3
"""Generate deterministic 24-hour NOAA peak-flux and RXFI targets.

The confirmed dataset contract assigns each frozen split timestamp ``t`` the
half-open future window [t, t + 24 hours).  Input split files are read only;
derived CSVs preserve every input row and its original order.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from inspect_noaa_xrs_flare_report import calculate_rxfi, finite_positive


DEFAULT_DATA_DIR = Path("/mnt/storage/surya/index_data")
DEFAULT_NOAA_CSV = Path("/tmp/rxfi-noaa/sci_xrsf-l2-flrpt_geo_s19950103_e20260913_v1-0-1.csv")
DEFAULT_DERIVED_DIR = Path("data/derived")
DEFAULT_OUTPUT_DIR = Path("outputs/target_generation_24h")
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
    "leaky_validation": "leaky_validation.csv",
}
TARGET_COLUMNS = (
    "original_row_index",
    "flare_count_24h",
    "max_flare_class",
    "max_peak_flux",
    "cumulative_peak_flux",
    "max_rxfi",
    "cumulative_rxfi",
    "max_peak_flare_id",
    "max_rxfi_flare_id",
)
SUMMARY_QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def sha256(path: Path) -> str:
    """Calculate an input-file checksum without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile of a sorted nonempty list."""
    index = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def read_noaa_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read NOAA records with a valid peak, retaining invalid-RXFI records for QC."""
    events: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_keys = 0
    invalid_peak_records = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            peak_time = datetime.fromisoformat(row["time"])
            # One extra day permits full [t, t + 24h) coverage for 2024-12-31.
            if not (2010 <= peak_time.year <= 2025):
                continue
            peak = finite_positive(row.get("xrsb_irrad"))
            if peak is None:
                invalid_peak_records += 1
                continue
            flare_id = row.get("flare_id", "").strip()
            source = row.get("xrsb_irrad_source", "").strip()
            stable_key = flare_id or f"{row['time']}|{source}"
            key = (stable_key, row["time"])
            if key in seen_keys:
                duplicate_keys += 1
                continue
            seen_keys.add(key)
            events.append(
                {
                    "time": peak_time,
                    "peak": peak,
                    "flare_id": stable_key,
                    "flare_class": row.get("flare_class", "").strip(),
                    "rxfi": calculate_rxfi(row.get("xrsb_irrad"), row.get("background_irrad")),
                }
            )
    events.sort(key=lambda event: (event["time"], event["flare_id"]))
    return events, {"duplicate_event_keys_skipped": duplicate_keys, "invalid_peak_records_skipped": invalid_peak_records}


def select_max(events: list[dict[str, Any]], value_name: str) -> dict[str, Any]:
    """Select maximum value, breaking ties by earliest peak time then stable ID."""
    return min(events, key=lambda event: (-float(event[value_name]), event["time"], event["flare_id"]))


def target_for_window(window: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    """Aggregate one canonical window and return target values plus quality counts."""
    valid_peaks = window
    valid_rxfi = [event for event in window if event["rxfi"] is not None]
    quality = {
        "events_with_valid_peak": len(valid_peaks),
        "events_invalid_rxfi": len(valid_peaks) - len(valid_rxfi),
        "peak_ties": 0,
        "rxfi_ties": 0,
    }
    if not valid_peaks:
        return {
            "flare_count_24h": 0,
            "max_flare_class": "FQ",
            "max_peak_flux": 0.0,
            "cumulative_peak_flux": 0.0,
            "max_rxfi": 0.0,
            "cumulative_rxfi": 0.0,
            "max_peak_flare_id": "",
            "max_rxfi_flare_id": "",
        }, quality
    max_peak = select_max(valid_peaks, "peak")
    quality["peak_ties"] = sum(event["peak"] == max_peak["peak"] for event in valid_peaks) - 1
    if valid_rxfi:
        max_rxfi = select_max(valid_rxfi, "rxfi")
        quality["rxfi_ties"] = sum(event["rxfi"] == max_rxfi["rxfi"] for event in valid_rxfi) - 1
        max_rxfi_value = float(max_rxfi["rxfi"])
        cumulative_rxfi = sum(float(event["rxfi"]) for event in valid_rxfi)
        max_rxfi_id = max_rxfi["flare_id"]
    else:
        max_rxfi_value = cumulative_rxfi = 0.0
        max_rxfi_id = ""
    return {
        "flare_count_24h": len(valid_peaks),
        "max_flare_class": max_peak["flare_class"],
        "max_peak_flux": max_peak["peak"],
        "cumulative_peak_flux": sum(event["peak"] for event in valid_peaks),
        "max_rxfi": max_rxfi_value,
        "cumulative_rxfi": cumulative_rxfi,
        "max_peak_flare_id": max_peak["flare_id"],
        "max_rxfi_flare_id": max_rxfi_id,
    }, quality


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict[str, float | int]:
    """Calculate requested continuous-target statistics, including zero counts."""
    sorted_values = sorted(values)
    return {
        "valid_count": len(sorted_values),
        "zero_count": sum(value == 0 for value in sorted_values),
        "missing_invalid_count": 0,
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "std": statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        "q1": percentile(sorted_values, 0.25),
        "q3": percentile(sorted_values, 0.75),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "maximum": sorted_values[-1],
    }


def svg_header(title: str, width: int = 960, height: int = 460) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}.title{font-size:18px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}.axis{stroke:#425466;stroke-width:1}.grid{stroke:#d8e1e8;stroke-width:1}.bar{fill:#2b6cb0}.point{fill:#2b6cb0;fill-opacity:.24}</style>',
        f'<text x="40" y="28" class="title">{title}</text>',
    ]


def save_svg(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def distributions_svg(path: Path, targets: dict[str, list[float]]) -> None:
    """Plot log10(1+x) histograms, retaining zero values in each distribution."""
    parts = svg_header("24-hour target distributions: log10(1 + target)")
    for index, (name, values) in enumerate(targets.items()):
        col, row = index % 2, index // 2
        x0, y0, width, height = 70 + col * 450, 70 + row * 190, 330, 110
        transformed = [math.log10(1 + value) for value in values]
        upper = max(transformed) or 1.0
        bins = [0] * 25
        for value in transformed:
            bins[min(24, int(25 * value / upper))] += 1
        peak = max(bins) or 1
        for bin_index, count in enumerate(bins):
            bar_width = width / len(bins)
            bar_height = height * count / peak
            parts.append(f'<rect x="{x0 + bin_index * bar_width:.1f}" y="{y0 + height - bar_height:.1f}" width="{bar_width - 1:.1f}" height="{bar_height:.1f}" class="bar"/>')
        parts.extend([
            f'<line x1="{x0}" y1="{y0+height}" x2="{x0+width}" y2="{y0+height}" class="axis"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+height}" class="axis"/>',
            f'<text x="{x0}" y="{y0-10}" class="label">{name}</text>',
            f'<text x="{x0}" y="{y0+height+16}" class="small">0</text>',
            f'<text x="{x0+width}" y="{y0+height+16}" text-anchor="end" class="small">{upper:.3g}</text>',
            f'<text x="{x0+width}" y="{y0-10}" text-anchor="end" class="small">zero n={sum(value == 0 for value in values):,}</text>',
        ])
    parts.append('<text x="480" y="440" text-anchor="middle" class="small">All split rows are included; zero/no-flare windows occupy the leftmost bin.</text>')
    save_svg(path, parts)


def scatter_svg(path: Path, peak_values: list[float], rxfi_values: list[float]) -> None:
    """Plot a deterministic sample of all rows in log10(1+x) coordinates."""
    width, height, left, top, chart_width, chart_height = 900, 460, 90, 65, 730, 310
    parts = svg_header("24-hour maximum peak flux versus maximum RXFI", width, height)
    x_values = [math.log10(1 + value) for value in peak_values]
    y_values = [math.log10(1 + value) for value in rxfi_values]
    xmax, ymax = max(x_values) or 1.0, max(y_values) or 1.0
    stride = max(1, len(x_values) // 18000)
    for index in range(0, len(x_values), stride):
        x = left + chart_width * x_values[index] / xmax
        y = top + chart_height * (1 - y_values[index] / ymax)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3" class="point"/>')
    parts.extend([
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" class="axis"/>',
        f'<text x="{left+chart_width/2}" y="{top+chart_height+36}" text-anchor="middle" class="label">log10(1 + max_peak_flux [W/m²])</text>',
        f'<text x="20" y="{top+chart_height/2}" transform="rotate(-90 20 {top+chart_height/2})" text-anchor="middle" class="label">log10(1 + max_rxfi)</text>',
        f'<text x="{left}" y="{top+chart_height+55}" class="small">Zero/no-flare rows: {sum(value == 0 for value in peak_values):,}</text>',
        f'<text x="{left+chart_width}" y="{top+chart_height+55}" text-anchor="end" class="small">Deterministic every-{stride}th-row display sample; all rows are summarized in tables.</text>',
    ])
    save_svg(path, parts)


def verify_half_open_window() -> None:
    """Assert that an event exactly at t+24h is excluded by bisect-left."""
    t0 = datetime(2024, 1, 1)
    times = [t0, t0 + timedelta(hours=23, minutes=59), t0 + timedelta(hours=24)]
    assert bisect.bisect_left(times, t0) == 0
    assert bisect.bisect_left(times, t0 + timedelta(hours=24)) == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--noaa-csv", type=Path, default=DEFAULT_NOAA_CSV)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.noaa_csv.is_file():
        raise SystemExit(f"NOAA CSV not found: {args.noaa_csv}")
    verify_half_open_window()
    events, noaa_qc = read_noaa_events(args.noaa_csv)
    event_times = [event["time"] for event in events]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    figure_targets: dict[str, list[float]] = {name: [] for name in ("max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi")}
    figure_peak: list[float] = []
    figure_rxfi: list[float] = []

    for split, filename in SPLIT_FILES.items():
        source_path = args.data_dir / filename
        before = sha256(source_path)
        with source_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            source_columns = reader.fieldnames or []
            source_rows = list(reader)
        generated_rows: list[dict[str, Any]] = []
        class_matches = class_mismatches = no_flare = multi_flare = peak_rxfi_different = both_max_ids = 0
        invalid_rxfi_events = peak_ties = rxfi_ties = 0
        mismatch_examples: list[dict[str, str]] = []
        for row_index, source_row in enumerate(source_rows):
            timestamp = datetime.fromisoformat(source_row["timestamp"])
            lo = bisect.bisect_left(event_times, timestamp)
            hi = bisect.bisect_left(event_times, timestamp + timedelta(hours=24))
            target, quality = target_for_window(events[lo:hi])
            generated = dict(source_row)
            generated["original_row_index"] = row_index
            generated.update(target)
            generated_rows.append(generated)
            label = source_row.get("max_goes_class", "")
            if target["max_flare_class"] == label:
                class_matches += 1
            else:
                class_mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append({"row_index": str(row_index), "timestamp": source_row["timestamp"], "existing_max_goes_class": label, "generated_max_flare_class": str(target["max_flare_class"]), "max_peak_flare_id": str(target["max_peak_flare_id"])})
            no_flare += target["flare_count_24h"] == 0
            multi_flare += target["flare_count_24h"] > 1
            both_ids_present = bool(target["max_peak_flare_id"]) and bool(target["max_rxfi_flare_id"])
            both_max_ids += both_ids_present
            peak_rxfi_different += both_ids_present and target["max_peak_flare_id"] != target["max_rxfi_flare_id"]
            invalid_rxfi_events += quality["events_invalid_rxfi"]
            peak_ties += quality["peak_ties"]
            rxfi_ties += quality["rxfi_ties"]
            for name in figure_targets:
                figure_targets[name].append(float(target[name]))
            figure_peak.append(float(target["max_peak_flux"]))
            figure_rxfi.append(float(target["max_rxfi"]))
        output_path = args.derived_dir / f"{split}_targets.csv"
        write_csv(output_path, source_columns + list(TARGET_COLUMNS), generated_rows)
        after = sha256(source_path)
        if before != after:
            raise RuntimeError(f"Source split changed while generating targets: {source_path}")
        with output_path.open(encoding="utf-8", newline="") as handle:
            output_rows = list(csv.DictReader(handle))
        if len(output_rows) != len(source_rows) or any(
            output_row["timestamp"] != source_row["timestamp"] or int(output_row["original_row_index"]) != index
            for index, (source_row, output_row) in enumerate(zip(source_rows, output_rows))
        ):
            raise RuntimeError(f"Row identity validation failed: {output_path}")
        validation_rows.append({"split": split, "source_rows": len(source_rows), "derived_rows": len(output_rows), "class_matches": class_matches, "class_mismatches": class_mismatches, "class_match_percent": 100 * class_matches / len(source_rows), "mismatch_examples": json.dumps(mismatch_examples), "sha256_before": before, "sha256_after": after, "row_order_and_timestamp_preserved": True})
        quality_rows.append({"split": split, "no_flare_windows": no_flare, "no_flare_percent": 100 * no_flare / len(source_rows), "multi_flare_windows": multi_flare, "multi_flare_percent": 100 * multi_flare / len(source_rows), "windows_with_both_max_ids": both_max_ids, "different_max_peak_and_rxfi_events": peak_rxfi_different, "different_max_peak_and_rxfi_percent_of_all_rows": 100 * peak_rxfi_different / len(source_rows), "different_max_peak_and_rxfi_percent_of_comparable_windows": 100 * peak_rxfi_different / both_max_ids if both_max_ids else 0.0, "events_invalid_rxfi_across_windows": invalid_rxfi_events, "peak_tie_candidates": peak_ties, "rxfi_tie_candidates": rxfi_ties})
        for target_name in figure_targets:
            stats = describe([float(row[target_name]) for row in generated_rows])
            summary_rows.append({"split": split, "target": target_name, **stats})

    write_csv(args.output_dir / "target_summary_by_split.csv", list(summary_rows[0]), summary_rows)
    write_csv(args.output_dir / "target_validation_against_existing_labels.csv", list(validation_rows[0]), validation_rows)
    write_csv(args.output_dir / "target_quality_control.csv", list(quality_rows[0]), quality_rows)
    metadata = {
        "noaa_csv": str(args.noaa_csv),
        "window": "[t, t + 24 hours)",
        "peak_event_eligibility": "finite positive xrsb_irrad",
        "rxfi_event_eligibility": "frozen calculate_rxfi(xrsb_irrad, background_irrad) is valid",
        "tie_break": "earliest peak time, then flare_id",
        "no_flare_convention": "all continuous targets are 0; event identifiers are empty; max_flare_class is FQ",
        "noaa_input_quality_control": noaa_qc,
    }
    (args.output_dir / "target_generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    distributions_svg(args.output_dir / "figure_24h_target_distributions.svg", figure_targets)
    scatter_svg(args.output_dir / "figure_max_peak_flux_vs_max_rxfi.svg", figure_peak, figure_rxfi)
    print(json.dumps({"derived_dir": str(args.derived_dir), "output_dir": str(args.output_dir), "events_loaded": len(events), "splits": validation_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
