#!/usr/bin/env python3
"""Audit physical XRS-B threshold exceedance in event-catalog FQ windows.

This is a read-only QC analysis.  FQ remains defined solely by the absence of
a NOAA flare-summary event in the confirmed [t, t + 24h) window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date

from refine_no_flare_max_peak_flux import PRIMARY_SCHEDULE, SOURCE_HASHES, parse_time, source_for

SPLITS = ("train", "validation", "test", "leaky_validation")
THRESHOLDS = {"A1": 1e-8, "B1": 1e-7, "C1": 1e-6, "M1": 1e-5, "X1": 1e-4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


class XrsSeries:
    """Read already-cached valid XRS-B observations, preserving timestamps."""
    def __init__(self, cache_dir: Path):
        self.cache_dir, self.loaded = cache_dir, {}

    def load(self, satellite: int, year: int) -> tuple[np.ndarray, np.ndarray] | None:
        key = (satellite, year)
        if key in self.loaded:
            return self.loaded[key]
        path = self.cache_dir / f"g{satellite}" / f"{year}.nc"
        if not path.is_file():
            return None
        with Dataset(path) as data:
            times = num2date(data["time"][:], data["time"].units,
                             only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            seconds = np.asarray([int(t.replace(tzinfo=timezone.utc).timestamp()) for t in times], dtype=np.int64)
            flux = np.asarray(data["xrsb_flux"][:], dtype=float)
            flags = np.asarray(data["xrsb_flag"][:])
        valid = np.isfinite(flux) & (flux > 0) & (flags == 0)
        result = (seconds[valid], flux[valid])
        self.loaded[key] = result
        return result

    def interval(self, satellite: int, start: datetime, end: datetime) -> tuple[np.ndarray, np.ndarray] | None:
        chunks: list[tuple[np.ndarray, np.ndarray]] = []
        for year in range(start.year, end.year + 1):
            item = self.load(satellite, year)
            if item is None:
                return None
            sec, flux = item
            lo, hi = np.searchsorted(sec, int(start.timestamp())), np.searchsorted(sec, int(end.timestamp()))
            chunks.append((sec[lo:hi], flux[lo:hi]))
        if not chunks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


def load_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                event_time = datetime.fromisoformat(row["time"]).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            events.append({"time": event_time, "flare_id": row.get("flare_id", ""),
                           "flare_class": row.get("flare_class", ""), "peak_flux": row.get("xrsb_irrad", "")})
    return sorted(events, key=lambda item: (item["time"], str(item["flare_id"])))


def nearest_event(events: list[dict[str, object]], event_seconds: list[int], point: datetime, direction: str) -> dict[str, object] | None:
    index = bisect_left(event_seconds, int(point.timestamp()))
    if direction == "before":
        index -= 1
    if 0 <= index < len(events):
        return events[index]
    return None


def event_fields(events: list[dict[str, object]], seconds: list[int], start: datetime, maximum: datetime) -> dict[str, object]:
    previous = nearest_event(events, seconds, start, "before")
    nearest_max_before = nearest_event(events, seconds, maximum, "before")
    nearest_max_after = nearest_event(events, seconds, maximum, "after")
    candidates = [item for item in (nearest_max_before, nearest_max_after) if item is not None]
    closest = min(candidates, key=lambda item: abs((item["time"] - maximum).total_seconds())) if candidates else None
    def describe(prefix: str, item: dict[str, object] | None, reference: datetime) -> dict[str, object]:
        if item is None:
            return {f"{prefix}_time": "", f"{prefix}_flare_id": "", f"{prefix}_class": "", f"{prefix}_peak_flux": "", f"{prefix}_delta_minutes": ""}
        return {f"{prefix}_time": item["time"].isoformat(), f"{prefix}_flare_id": item["flare_id"],
                f"{prefix}_class": item["flare_class"], f"{prefix}_peak_flux": item["peak_flux"],
                f"{prefix}_delta_minutes": fmt((item["time"] - reference).total_seconds() / 60)}
    result = describe("previous_event", previous, start)
    result.update(describe("nearest_event_to_max", closest, maximum))
    prior_decay = previous is not None and 0 <= (start - previous["time"]).total_seconds() <= 6 * 3600
    near_outside = closest is not None and abs((closest["time"] - maximum).total_seconds()) <= 6 * 3600 and not (start <= closest["time"] < start + timedelta(hours=24))
    result["previous_flare_within_6h_before_window"] = str(prior_decay).lower()
    result["near_max_event_outside_exact_window_within_6h"] = str(near_outside).lower()
    return result


def shape_label(seconds: np.ndarray, flux: np.ndarray, maximum: datetime, events: dict[str, object]) -> tuple[str, str]:
    """Conservative heuristic only; it never assigns physical causality."""
    if len(flux) < 90:
        return "data_artifact_or_gap", "fewer than 90 valid 1-minute values in ±60 min"
    peak = float(np.max(flux)); median = float(np.median(flux))
    peak_index = int(np.argmax(flux)); before, after = flux[:peak_index], flux[peak_index + 1:]
    close_before = float(np.median(before[-20:])) if len(before) >= 20 else peak
    close_after = float(np.median(after[:20])) if len(after) >= 20 else peak
    if events["previous_flare_within_6h_before_window"] == "true" and peak_index < 20:
        return "decay_from_prior_flare", "prior catalog event is within 6h before window and local maximum is near window start"
    if median >= 0.5 * peak:
        return "elevated_background", "local median is at least half the local maximum"
    if close_before < 0.5 * peak and close_after < 0.5 * peak:
        return "isolated_short_rise", "both 20-minute neighborhoods are below half the local maximum"
    if close_before >= 0.5 * peak or close_after >= 0.5 * peak:
        return "broad_gradual_structure", "at least one 20-minute neighborhood remains above half the maximum"
    return "unclear", "heuristic criteria do not isolate a pattern"


def source_interval(series: XrsSeries, start: datetime, end: datetime) -> tuple[int | None, str, tuple[np.ndarray, np.ndarray] | None]:
    primary, secondary = source_for(start)
    values = series.interval(primary, start, end)
    if values is not None and len(values[0]):
        return primary, "primary", values
    values = series.interval(secondary, start, end) if secondary else None
    return (secondary, "secondary", values) if values is not None and len(values[0]) else (None, "missing", None)


def pct(numerator: int, denominator: int) -> float | None:
    return 100 * numerator / denominator if denominator else None


def summary_row(name: str, samples: list[dict[str, object]]) -> dict[str, object]:
    values = [float(item["max_peak_flux"]) for item in samples if item["max_peak_flux"] is not None]
    row: dict[str, object] = {"group": name, "fq_count": len(samples), "valid_count": len(values), "missing_count": len(samples) - len(values)}
    for label, threshold in THRESHOLDS.items():
        count = sum(value >= threshold for value in values)
        row[f"{label}_count"], row[f"{label}_percent_valid"] = count, pct(count, len(values))
    if values:
        row.update({"min": min(values), "q1": float(np.quantile(values, .25)), "median": float(np.median(values)), "mean": float(np.mean(values)), "q3": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": max(values)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split-dir", type=Path, default=Path("/mnt/storage/surya/index_data"))
    parser.add_argument("--derived-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/noaa_xrs_avg1m"))
    parser.add_argument("--event-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fq_xrs_threshold_exceedance_qc"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    before_hashes = {split: sha256(args.derived_dir / f"{split}_targets.csv") for split in SPLITS}
    source_hashes = {split: sha256(args.source_split_dir / f"{split}.csv") for split in SPLITS}
    if source_hashes != SOURCE_HASHES:
        raise RuntimeError("A frozen split SHA-256 hash differs from the confirmed contract")
    events = load_events(args.event_csv); event_seconds = [int(item["time"].timestamp()) for item in events]
    series = XrsSeries(args.cache_dir)
    fq: list[dict[str, object]] = []
    for split in SPLITS:
        with (args.derived_dir / f"{split}_targets.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["max_flare_class"] != "FQ":
                    continue
                value = float(row["max_peak_flux"]) if row["max_peak_flux"].strip() else None
                fq.append({"split": split, "timestamp": parse_time(row["timestamp"]), "year": parse_time(row["timestamp"]).year,
                           "max_peak_flux": value, "valid_count_24h": row.get("xrsb_1min_valid_count_24h", ""),
                           "coverage_fraction_24h": row.get("xrsb_1min_coverage_fraction_24h", "")})
    overall = summary_row("overall", fq)
    split_rows = [summary_row(split, [item for item in fq if item["split"] == split]) for split in SPLITS]
    years = sorted({int(item["year"]) for item in fq})
    year_rows = [summary_row(str(year), [item for item in fq if item["year"] == year]) for year in years]
    write_csv(args.output_dir / "fq_threshold_exceedance_overall.csv", [overall])
    write_csv(args.output_dir / "fq_threshold_exceedance_by_split.csv", split_rows)
    write_csv(args.output_dir / "fq_threshold_exceedance_by_year.csv", year_rows)
    high = [item for item in fq if item["max_peak_flux"] is not None and float(item["max_peak_flux"]) >= THRESHOLDS["C1"]]
    high.sort(key=lambda item: (item["timestamp"], item["split"]))
    audit_rows: list[dict[str, object]] = []; local_cases: list[dict[str, object]] = []; local_series: list[dict[str, object]] = []
    chosen = high[:10] + [item for item in high if float(item["max_peak_flux"]) >= THRESHOLDS["M1"] and item not in high[:10]]
    for index, item in enumerate(high):
        start = item["timestamp"]; end = start + timedelta(hours=24)
        satellite, source_role, data = source_interval(series, start, end)
        if data is None:
            raise RuntimeError(f"Cached XRS series unavailable for high-flux window {start}")
        seconds, flux = data; peak_index = int(np.argmax(flux)); maximum = datetime.fromtimestamp(int(seconds[peak_index]), tz=timezone.utc)
        details = event_fields(events, event_seconds, start, maximum)
        audit = {"split": item["split"], "timestamp": start.isoformat(), "year": item["year"], "max_peak_flux": fmt(float(item["max_peak_flux"])),
                 "log10_max_peak_flux": fmt(math.log10(float(item["max_peak_flux"]))), "xrsb_max_timestamp": maximum.isoformat(),
                 "source_satellite": satellite or "", "source_role": source_role, "valid_count_24h": item["valid_count_24h"],
                 "coverage_fraction_24h": item["coverage_fraction_24h"]}
        audit.update(details); audit_rows.append(audit)
        if item in chosen:
            local = series.interval(satellite, maximum - timedelta(minutes=60), maximum + timedelta(minutes=61)) if satellite else None
            local_seconds, local_flux = local if local is not None else (np.empty(0, dtype=int), np.empty(0))
            label, rationale = shape_label(local_seconds, local_flux, maximum, details)
            case_id = f"case_{len(local_cases)+1:03d}"
            local_cases.append({"case_id": case_id, **audit, "heuristic_shape": label, "heuristic_rationale": rationale,
                                "local_valid_count": len(local_flux)})
            for sec, value in zip(local_seconds, local_flux):
                local_series.append({"case_id": case_id, "timestamp": datetime.fromtimestamp(int(sec), tz=timezone.utc).isoformat(), "xrsb_flux": fmt(float(value))})
    fields = ["split", "timestamp", "year", "max_peak_flux", "log10_max_peak_flux", "xrsb_max_timestamp", "source_satellite", "source_role", "valid_count_24h", "coverage_fraction_24h", "previous_event_time", "previous_event_flare_id", "previous_event_class", "previous_event_peak_flux", "previous_event_delta_minutes", "nearest_event_to_max_time", "nearest_event_to_max_flare_id", "nearest_event_to_max_class", "nearest_event_to_max_peak_flux", "nearest_event_to_max_delta_minutes", "previous_flare_within_6h_before_window", "near_max_event_outside_exact_window_within_6h"]
    write_csv(args.output_dir / "fq_c1_or_higher_audit.csv", audit_rows, fields)
    write_csv(args.output_dir / "fq_m1_or_higher_audit.csv", [row for row in audit_rows if float(row["max_peak_flux"]) >= THRESHOLDS["M1"]], fields)
    write_csv(args.output_dir / "local_timeseries_case_summary.csv", local_cases)
    write_csv(args.output_dir / "local_timeseries_1min.csv", local_series, ["case_id", "timestamp", "xrsb_flux"])
    coverage_rows = []
    for label, threshold in (("C1_or_higher", THRESHOLDS["C1"]), ("M1_or_higher", THRESHOLDS["M1"])):
        subset = [row for row in audit_rows if float(row["max_peak_flux"]) >= threshold]
        cov = [float(row["coverage_fraction_24h"]) for row in subset]
        coverage_rows.append({"threshold_group": label, "count": len(subset), "coverage_median": float(np.median(cov)), "coverage_min": min(cov),
                              "primary_source_count": sum(row["source_role"] == "primary" for row in subset), "secondary_source_count": sum(row["source_role"] == "secondary" for row in subset),
                              "satellite_distribution": json.dumps(dict(sorted(Counter(str(row["source_satellite"]) for row in subset).items())))})
    write_csv(args.output_dir / "high_flux_coverage_quality.csv", coverage_rows)
    shape_rows = [{"heuristic_shape": key, "count": value} for key, value in sorted(Counter(row["heuristic_shape"] for row in local_cases).items())]
    write_csv(args.output_dir / "local_timeseries_shape_summary.csv", shape_rows, ["heuristic_shape", "count"])
    valid_flux = np.asarray([float(item["max_peak_flux"]) for item in fq if item["max_peak_flux"] is not None])
    fig, ax = plt.subplots(figsize=(8, 4.5)); bins = np.logspace(-9, -3, 70); ax.hist(valid_flux, bins=bins, color="#2b6cb0")
    for label, threshold in THRESHOLDS.items(): ax.axvline(threshold, linestyle="--", linewidth=1, label=label)
    ax.set(xscale="log", xlabel="FQ 24h maximum XRS-B flux (W/m²)", ylabel="FQ windows"); ax.legend(ncol=5, fontsize=8); fig.tight_layout(); fig.savefig(args.output_dir / "figure_fq_flux_distribution_thresholds.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(SPLITS)); width = .16
    for offset, (label, _) in enumerate(THRESHOLDS.items()): ax.bar(x + (offset - 2) * width, [float(row[f"{label}_percent_valid"] or 0) for row in split_rows], width, label=label)
    ax.set(xticks=x, xticklabels=SPLITS, ylabel="FQ valid windows exceeding threshold (%)"); ax.legend(ncol=5, fontsize=8); fig.tight_layout(); fig.savefig(args.output_dir / "figure_fq_threshold_exceedance_by_split.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.plot(years, [float(row["B1_percent_valid"] or 0) for row in year_rows], marker="o", label="B1")
    ax.plot(years, [float(row["C1_percent_valid"] or 0) for row in year_rows], marker="o", label="C1"); ax.plot(years, [float(row["M1_percent_valid"] or 0) for row in year_rows], marker="o", label="M1")
    ax.set(xlabel="Window start year", ylabel="FQ valid windows exceeding threshold (%)"); ax.legend(); fig.tight_layout(); fig.savefig(args.output_dir / "figure_fq_threshold_exceedance_by_year.svg"); plt.close(fig)
    after_hashes = {split: sha256(args.derived_dir / f"{split}_targets.csv") for split in SPLITS}
    if before_hashes != after_hashes:
        raise RuntimeError("QC analysis modified derived target files")
    metadata = {"purpose": "read-only FQ XRS threshold-exceedance QC", "thresholds_w_m2": THRESHOLDS, "source_split_sha256": source_hashes,
                "derived_target_sha256_before": before_hashes, "derived_target_sha256_after": after_hashes, "target_files_unchanged": True,
                "fq_selection": "max_flare_class == FQ", "event_catalog": str(args.event_csv), "event_catalog_sha256": sha256(args.event_csv),
                "overall": overall, "c1_or_higher_count": len(audit_rows), "m1_or_higher_count": sum(float(row["max_peak_flux"]) >= THRESHOLDS["M1"] for row in audit_rows)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
