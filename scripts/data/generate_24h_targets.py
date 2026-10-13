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
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from inspect_noaa_xrs_flare_report import calculate_rxfi, finite_positive
except ModuleNotFoundError:
    from scripts.data.inspect_noaa_xrs_flare_report import calculate_rxfi, finite_positive

try:
    from scripts.background.paper_background.calculation import (
        mask_merged_flare_intervals,
        merge_flare_intervals,
        valid_measurement_mask,
    )
    from scripts.background.paper_background.noaa import (
        find_cached_input,
        primary_secondary_satellite_for,
        read_l2_year,
    )
except ModuleNotFoundError:
    from paper_background.calculation import mask_merged_flare_intervals, merge_flare_intervals, valid_measurement_mask
    from paper_background.noaa import find_cached_input, primary_secondary_satellite_for, read_l2_year


DEFAULT_DATA_DIR = Path("/mnt/storage/surya/index_data")
DEFAULT_NOAA_CSV = Path("/tmp/rxfi-noaa/sci_xrsf-l2-flrpt_geo_s19950103_e20260913_v1-0-1.csv")
DEFAULT_DERIVED_DIR = Path("data/derived")
DEFAULT_OUTPUT_DIR = Path("outputs/target_generation_24h")
DEFAULT_XRS_CACHE_DIR = Path("data/raw/noaa_paper_background")
DEFAULT_XRS_INPUT_LOCK = Path("configs/paper_background_sensitivity.inputs.lock.json")
XRS_WINDOW_SECONDS = 24 * 60 * 60
FQ_BACKGROUND_OFFSET_SECONDS = 5 * 60
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
    "max_peak_time_utc",
    "max_peak_satellite",
    "max_peak_flux_source",
    "fq_background_flux",
    "fq_background_valid_count",
    "fq_background_masked_count",
    "fq_rxfi_status",
)
SUMMARY_QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def sha256(path: Path) -> str:
    """Calculate an input-file checksum without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_seconds(value: datetime) -> int:
    """Convert a catalog/split timestamp, interpreted as UTC, to epoch seconds."""
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return int(aware.timestamp())


class XrsSeriesCache:
    """Read checksum-pinned, science-quality XRS-B annual files on demand."""

    def __init__(self, cache_dir: Path, lock_path: Path):
        self.cache_dir = cache_dir
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.version = lock["xrs_l2"]["version"]
        self.locked_files = {
            (int(item["satellite"]), int(item["year"])): item
            for item in lock["xrs_l2"]["files"]
        }
        self.series: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.loaded_inputs: dict[tuple[int, int], dict[str, Any]] = {}

    def load(self, satellite: int, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (satellite, year)
        if key in self.series:
            return self.series[key]
        locked = self.locked_files.get(key)
        if locked is None:
            raise FileNotFoundError(f"XRS input lock has no entry for GOES-{satellite}/{year}")
        path, record = find_cached_input(
            self.cache_dir,
            satellite,
            year,
            self.version,
            expected_sha256=locked["sha256"],
            expected_size=locked.get("size_bytes"),
        )
        seconds, flux, flags, _ = read_l2_year(path)
        valid = valid_measurement_mask(flux, flags)
        result = (seconds[valid], flux[valid], flags[valid])
        self.series[key] = result
        self.loaded_inputs[key] = {
            "satellite": satellite,
            "year": year,
            "path": str(path),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "legacy_layout": record["legacy_layout"],
        }
        return result

    def window(self, satellite: int, start: datetime, end: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Return finite positive, quality-zero samples in [start, end)."""
        start_seconds, end_seconds = utc_seconds(start), utc_seconds(end)
        last_included_year = datetime.fromtimestamp(end_seconds - 1, timezone.utc).year
        pieces: list[tuple[np.ndarray, np.ndarray]] = []
        for year in range(datetime.fromtimestamp(start_seconds, timezone.utc).year, last_included_year + 1):
            seconds, flux, _ = self.load(satellite, year)
            left = int(np.searchsorted(seconds, start_seconds, side="left"))
            right = int(np.searchsorted(seconds, end_seconds, side="left"))
            pieces.append((seconds[left:right], flux[left:right]))
        if not pieces:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        return np.concatenate([part[0] for part in pieces]), np.concatenate([part[1] for part in pieces])


def fq_window_target(
    cache: XrsSeriesCache,
    timestamp: datetime,
    primary_satellite: int,
    secondary_satellite: int | None,
    merged_flare_intervals: tuple[np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build one FQ regression target from its window maximum and local B50."""
    window_end = timestamp + timedelta(hours=24)
    seconds, flux = cache.window(primary_satellite, timestamp, window_end)
    satellite = primary_satellite
    if len(seconds) == 0 and secondary_satellite is not None:
        try:
            seconds, flux = cache.window(secondary_satellite, timestamp, window_end)
            satellite = secondary_satellite
        except FileNotFoundError:
            # Some secondary-satellite annual files are not required by the
            # pinned event-background cohort and therefore are absent locally.
            # The refinement convention treats an unavailable fallback as a
            # missing XRS window rather than silently substituting another source.
            seconds, flux = np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    if len(seconds) == 0:
        return {
            "max_peak_flux": None,
            "cumulative_peak_flux": None,
            "max_rxfi": None,
            "cumulative_rxfi": None,
            "max_peak_time_utc": "",
            "max_peak_satellite": f"GOES-{satellite}",
            "max_peak_flux_source": "missing_xrsb_1min",
            "fq_background_flux": None,
            "fq_background_valid_count": 0,
            "fq_background_masked_count": 0,
            "fq_rxfi_status": "missing_window_xrsb",
        }, {"fq_xrs_missing": 1, "fq_negative_rxfi_excluded": 0, "fq_background_missing": 0}

    max_index = int(np.argmax(flux))
    max_seconds = int(seconds[max_index])
    max_flux = float(format(float(flux[max_index]), ".12g"))
    peak_time = datetime.fromtimestamp(max_seconds, timezone.utc)
    background_end = peak_time - timedelta(seconds=FQ_BACKGROUND_OFFSET_SECONDS)
    background_start = background_end - timedelta(seconds=XRS_WINDOW_SECONDS)
    background_seconds, background_flux = cache.window(satellite, background_start, background_end)
    masked = mask_merged_flare_intervals(background_seconds, merged_flare_intervals)
    retained_values = background_flux[~masked]
    retained_count = int(len(retained_values))
    if retained_count == 0:
        rxfi = None
        rxfi_status = "missing_background"
    else:
        lower_count = int(math.ceil(retained_count * 0.5))
        lower_values = np.partition(retained_values, lower_count - 1)[:lower_count]
        background = float(format(float(np.mean(lower_values)), ".12g"))
        rxfi = float(format((max_flux - background) / background, ".12g"))
        if rxfi < 0:
            rxfi, rxfi_status = 0.0, "negative_excluded"
        else:
            rxfi_status = "ok"
    return {
        "max_peak_flux": max_flux,
        "cumulative_peak_flux": max_flux,
        "max_rxfi": rxfi,
        "cumulative_rxfi": rxfi,
        "max_peak_time_utc": peak_time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_peak_satellite": f"GOES-{satellite}",
        "max_peak_flux_source": "xrsb_1min_fq_window_max",
        "fq_background_flux": background if retained_count else None,
        "fq_background_valid_count": retained_count,
        "fq_background_masked_count": int(masked.sum()),
        "fq_rxfi_status": rxfi_status,
    }, {
        "fq_xrs_missing": 0,
        "fq_negative_rxfi_excluded": int(rxfi_status == "negative_excluded"),
        "fq_background_missing": int(rxfi_status == "missing_background"),
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile of a sorted nonempty list."""
    index = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def normalized_event_time(value: str) -> str:
    """Normalize source timestamps to the minute-resolution event key format."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def read_lower50_rxfi(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Read event-level lower-50% RXFI keyed by catalog peak time and flare ID."""
    results: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flare_id = row.get("flare_id", "").strip()
            key = (normalized_event_time(row["time"]), flare_id)
            if key in results:
                raise ValueError(f"Duplicate lower-50% event key: {key}")
            rxfi_text = row.get("rxfi_lower_fractional_excess", "").strip()
            status = row.get("lower_status", "").strip()
            peak_text = row.get("peak_flux_W_m2", "").strip()
            background_text = row.get("background_lower_fraction_W_m2", "").strip()
            if status != "ok" or not rxfi_text or not peak_text or not background_text:
                continue
            rxfi, peak, background = float(rxfi_text), float(peak_text), float(background_text)
            if math.isfinite(rxfi) and math.isfinite(peak) and peak > 0 and math.isfinite(background) and background > 0:
                calculated_rxfi = (peak - background) / background
                if not math.isclose(rxfi, calculated_rxfi, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"Stored lower-50% RXFI disagrees with its peak/background for {key}")
                results[key] = (rxfi, peak)
    return results


def read_noaa_events(
    path: Path,
    lower50_rxfi: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[tuple[int, int, str]]]:
    """Read NOAA records with a valid peak, retaining invalid-RXFI records for QC."""
    events: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    flare_intervals: list[tuple[int, int, str]] = []
    duplicate_keys = 0
    invalid_peak_records = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_text, end_text = row.get("start_time", ""), row.get("end_time", "")
            if start_text and end_text:
                start_time = datetime.fromisoformat(start_text)
                end_time = datetime.fromisoformat(end_text)
                if end_time >= start_time:
                    flare_intervals.append((utc_seconds(start_time), utc_seconds(end_time), row.get("flare_id", "").strip()))
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
            if lower50_rxfi is None:
                rxfi = calculate_rxfi(row.get("xrsb_irrad"), row.get("background_irrad"))
            else:
                match = lower50_rxfi.get((normalized_event_time(row["time"]), flare_id))
                if match is not None and not math.isclose(peak, match[1], rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"Lower-50% peak flux disagrees with NOAA catalog for {flare_id} at {row['time']}")
                rxfi = match[0] if match is not None else None
            events.append(
                {
                    "time": peak_time,
                    "peak": peak,
                    "flare_id": stable_key,
                    "flare_class": row.get("flare_class", "").strip(),
                    "satellite": source,
                    "rxfi": rxfi,
                }
            )
    events.sort(key=lambda event: (event["time"], event["flare_id"]))
    return events, {"duplicate_event_keys_skipped": duplicate_keys, "invalid_peak_records_skipped": invalid_peak_records}, flare_intervals


def select_max(events: list[dict[str, Any]], value_name: str) -> dict[str, Any]:
    """Select maximum value, breaking ties by earliest peak time then stable ID."""
    return min(events, key=lambda event: (-float(event[value_name]), event["time"], event["flare_id"]))


def target_for_window(window: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    """Aggregate one canonical window and return target values plus quality counts."""
    valid_peaks = window
    valid_rxfi = [event for event in window if event["rxfi"] is not None and event["rxfi"] >= 0]
    invalid_rxfi_events = sum(event["rxfi"] is None for event in window)
    negative_rxfi_events = sum(event["rxfi"] is not None and event["rxfi"] < 0 for event in window)
    quality = {
        "events_with_valid_peak": len(valid_peaks),
        "events_invalid_rxfi": invalid_rxfi_events,
        "events_excluded_negative_rxfi": negative_rxfi_events,
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
            "max_peak_time_utc": "",
            "max_peak_satellite": "",
            "max_peak_flux_source": "noaa_flare_report",
            "fq_background_flux": None,
            "fq_background_valid_count": 0,
            "fq_background_masked_count": 0,
            "fq_rxfi_status": "",
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
        "max_peak_time_utc": max_peak["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "max_peak_satellite": max_peak["satellite"],
        "max_peak_flux_source": "noaa_flare_report",
        "fq_background_flux": None,
        "fq_background_valid_count": 0,
        "fq_background_masked_count": 0,
        "fq_rxfi_status": "",
    }, quality


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float | None]) -> dict[str, float | int]:
    """Calculate requested continuous-target statistics, including zero counts."""
    sorted_values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not sorted_values:
        return {"valid_count": 0, "zero_count": 0, "missing_invalid_count": len(values)}
    return {
        "valid_count": len(sorted_values),
        "zero_count": sum(value == 0 for value in sorted_values),
        "missing_invalid_count": len(values) - len(sorted_values),
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


def distributions_svg(path: Path, targets: dict[str, list[float | None]]) -> None:
    """Plot log10(1+x) histograms, retaining zero values in each distribution."""
    parts = svg_header("24-hour target distributions: log10(1 + target)")
    for index, (name, values) in enumerate(targets.items()):
        values = [value for value in values if value is not None and math.isfinite(value)]
        if not values:
            continue
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
    parts.append('<text x="480" y="440" text-anchor="middle" class="small">All valid split targets are included; FQ rows use XRS-B window maxima.</text>')
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
        f'<text x="{left}" y="{top+chart_height+55}" class="small">Valid target pairs plotted: {len(peak_values):,}; FQ rows use XRS-B window maxima.</text>',
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
    parser.add_argument("--rxfi-method", choices=("noaa", "lower50"), default="noaa")
    parser.add_argument("--lower50-backgrounds-csv", type=Path)
    parser.add_argument("--xrs-cache-dir", type=Path, default=DEFAULT_XRS_CACHE_DIR)
    parser.add_argument("--xrs-input-lock", type=Path, default=DEFAULT_XRS_INPUT_LOCK)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.noaa_csv.is_file():
        raise SystemExit(f"NOAA CSV not found: {args.noaa_csv}")
    if args.rxfi_method == "lower50" and (args.lower50_backgrounds_csv is None or not args.lower50_backgrounds_csv.is_file()):
        raise SystemExit("--rxfi-method lower50 requires an existing --lower50-backgrounds-csv")
    if args.rxfi_method == "noaa" and args.lower50_backgrounds_csv is not None:
        raise SystemExit("--lower50-backgrounds-csv is only valid with --rxfi-method lower50")
    if not args.xrs_input_lock.is_file():
        raise SystemExit(f"XRS input lock not found: {args.xrs_input_lock}")
    verify_half_open_window()
    lower50_rxfi = read_lower50_rxfi(args.lower50_backgrounds_csv) if args.rxfi_method == "lower50" else None
    events, noaa_qc, flare_intervals = read_noaa_events(args.noaa_csv, lower50_rxfi)
    event_times = [event["time"] for event in events]
    merged_flare_intervals = merge_flare_intervals(flare_intervals)
    xrs_cache = XrsSeriesCache(args.xrs_cache_dir, args.xrs_input_lock)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    figure_targets: dict[str, list[float | None]] = {name: [] for name in ("max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi")}
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
        invalid_rxfi_events = negative_rxfi_events = peak_ties = rxfi_ties = 0
        fq_windows = fq_windows_with_xrs = fq_windows_missing_xrs = 0
        fq_negative_rxfi_excluded = fq_background_missing = 0
        fq_rxfi_positive = fq_rxfi_zero = fq_rxfi_missing = 0
        mismatch_examples: list[dict[str, str]] = []
        for row_index, source_row in enumerate(source_rows):
            timestamp = datetime.fromisoformat(source_row["timestamp"])
            lo = bisect.bisect_left(event_times, timestamp)
            hi = bisect.bisect_left(event_times, timestamp + timedelta(hours=24))
            target, quality = target_for_window(events[lo:hi])
            if target["max_flare_class"] == "FQ":
                fq_windows += 1
                primary, secondary = primary_secondary_satellite_for(timestamp)
                fq_target, fq_quality = fq_window_target(
                    xrs_cache, timestamp, primary, secondary, merged_flare_intervals
                )
                target.update(fq_target)
                fq_windows_with_xrs += int(fq_quality["fq_xrs_missing"] == 0)
                fq_windows_missing_xrs += fq_quality["fq_xrs_missing"]
                fq_negative_rxfi_excluded += fq_quality["fq_negative_rxfi_excluded"]
                fq_background_missing += fq_quality["fq_background_missing"]
                fq_rxfi_missing += int(target["max_rxfi"] is None)
                fq_rxfi_zero += int(target["max_rxfi"] == 0)
                fq_rxfi_positive += int(target["max_rxfi"] is not None and target["max_rxfi"] > 0)
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
            negative_rxfi_events += quality["events_excluded_negative_rxfi"]
            peak_ties += quality["peak_ties"]
            rxfi_ties += quality["rxfi_ties"]
            for name in figure_targets:
                figure_targets[name].append(target[name])
            if target["max_peak_flux"] is not None and target["max_rxfi"] is not None:
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
        quality_rows.append({
            "split": split,
            "no_flare_windows": no_flare,
            "no_flare_percent": 100 * no_flare / len(source_rows),
            "multi_flare_windows": multi_flare,
            "multi_flare_percent": 100 * multi_flare / len(source_rows),
            "windows_with_both_max_ids": both_max_ids,
            "different_max_peak_and_rxfi_events": peak_rxfi_different,
            "different_max_peak_and_rxfi_percent_of_all_rows": 100 * peak_rxfi_different / len(source_rows),
            "different_max_peak_and_rxfi_percent_of_comparable_windows": 100 * peak_rxfi_different / both_max_ids if both_max_ids else 0.0,
            "events_invalid_rxfi_across_windows": invalid_rxfi_events,
            "events_excluded_negative_rxfi_across_windows": negative_rxfi_events,
            "peak_tie_candidates": peak_ties,
            "rxfi_tie_candidates": rxfi_ties,
            "fq_windows": fq_windows,
            "fq_windows_with_valid_xrs_max": fq_windows_with_xrs,
            "fq_windows_missing_xrs_max": fq_windows_missing_xrs,
            "fq_negative_rxfi_excluded": fq_negative_rxfi_excluded,
            "fq_windows_missing_background": fq_background_missing,
            "fq_rxfi_positive_count": fq_rxfi_positive,
            "fq_rxfi_zero_count": fq_rxfi_zero,
            "fq_rxfi_missing_count": fq_rxfi_missing,
        })
        for target_name in figure_targets:
            stats = describe([
                None if row[target_name] is None else float(row[target_name])
                for row in generated_rows
            ])
            summary_rows.append({"split": split, "target": target_name, **stats})

    write_csv(args.output_dir / "target_summary_by_split.csv", list(summary_rows[0]), summary_rows)
    write_csv(args.output_dir / "target_validation_against_existing_labels.csv", list(validation_rows[0]), validation_rows)
    write_csv(args.output_dir / "target_quality_control.csv", list(quality_rows[0]), quality_rows)
    metadata = {
        "noaa_csv": str(args.noaa_csv),
        "noaa_csv_sha256": sha256(args.noaa_csv),
        "rxfi_method": args.rxfi_method,
        "lower50_backgrounds_csv": str(args.lower50_backgrounds_csv) if args.lower50_backgrounds_csv else None,
        "lower50_backgrounds_csv_sha256": sha256(args.lower50_backgrounds_csv) if args.lower50_backgrounds_csv else None,
        "window": "[t, t + 24 hours)",
        "peak_event_eligibility": "finite positive xrsb_irrad",
        "rxfi_event_eligibility": (
            "finite lower-50% RXFI from event_backgrounds.csv matched by peak time and flare_id; negative RXFI events are excluded from max_rxfi and cumulative_rxfi"
            if args.rxfi_method == "lower50"
            else "calculate_rxfi(xrsb_irrad, background_irrad) is finite and non-negative"
        ),
        "events_without_valid_rxfi": sum(event["rxfi"] is None for event in events),
        "events_excluded_negative_rxfi": sum(event["rxfi"] is not None and event["rxfi"] < 0 for event in events),
        "split_inputs": [
            {"split": row["split"], "source_rows": row["source_rows"], "sha256": row["sha256_before"]}
            for row in validation_rows
        ],
        "tie_break": "earliest peak time, then flare_id",
        "no_flare_convention": "FQ uses the maximum valid science-quality XRS-B one-minute sample in [t,t+24h) as one pseudo-event; max and cumulative flux targets equal that value; RXFI uses lower-50% background in [a-24h-5m,a-5m), with negative RXFI excluded; no valid XRS peak yields missing continuous targets",
        "fq_rxfi_method": "lower-50% mean of eligible same-satellite XRS-B samples in [a-24h-5m,a-5m); negative values excluded to zero",
        "fq_background_offset_seconds_before_maximum": FQ_BACKGROUND_OFFSET_SECONDS,
        "fq_background_window": "[a-24h-5m, a-5m)",
        "xrs_cache_dir": str(args.xrs_cache_dir),
        "xrs_input_lock": str(args.xrs_input_lock),
        "xrs_input_lock_sha256": sha256(args.xrs_input_lock),
        "xrs_inputs_used": [xrs_cache.loaded_inputs[key] for key in sorted(xrs_cache.loaded_inputs)],
        "noaa_input_quality_control": noaa_qc,
    }
    (args.output_dir / "target_generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    distributions_svg(args.output_dir / "figure_24h_target_distributions.svg", figure_targets)
    scatter_svg(args.output_dir / "figure_max_peak_flux_vs_max_rxfi.svg", figure_peak, figure_rxfi)
    print(json.dumps({"derived_dir": str(args.derived_dir), "output_dir": str(args.output_dir), "events_loaded": len(events), "splits": validation_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
