#!/usr/bin/env python3
"""Plot the June 14, 2010 flare example used to explain the paper cap."""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv"
XRS_FILE = ROOT / "data/raw/noaa_paper_background/xrs_l2/g14/2010.nc"
EVENT_RESULTS = ROOT / "outputs/paper_background_sensitivity/event_backgrounds.csv"
EVENT_ID = "201006140015"
DEFAULT_OUTPUT = ROOT / "outputs/paper_background_sensitivity/figures/capped_background_example_20100614"
UTC = timezone.utc


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def read_event() -> dict[str, str]:
    with EVENT_RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        event = next((row for row in rows if row["flare_id"] == EVENT_ID), None)
    if event is None:
        raise ValueError(f"Event {EVENT_ID} is absent from {EVENT_RESULTS}")
    return event


def read_catalog_intervals(start: datetime, end: datetime) -> list[tuple[datetime, datetime, str]]:
    intervals = []
    with CATALOG.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("flare_id", "").strip() == EVENT_ID:
                continue
            if not row.get("start_time") or not row.get("end_time"):
                continue
            flare_start, flare_end = parse_time(row["start_time"]), parse_time(row["end_time"])
            if flare_end >= start and flare_start < end:
                intervals.append((flare_start, flare_end, row.get("flare_class", "")))
    return intervals


def read_xrs_window(start: datetime, end: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(XRS_FILE) as dataset:
        time_variable = dataset.variables["time"]
        dates = num2date(
            time_variable[:],
            time_variable.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        seconds = np.asarray(
            [int(value.replace(tzinfo=UTC).timestamp()) for value in dates], dtype=np.int64
        )
        flux = np.asarray(dataset.variables["xrsb_flux"][:], dtype=float)
        flags = np.asarray(dataset.variables["xrsb_flag"][:])
    left = int(np.searchsorted(seconds, int(start.timestamp()), side="left"))
    right = int(np.searchsorted(seconds, int(end.timestamp()), side="left"))
    return seconds[left:right], flux[left:right], flags[left:right]


def make_figure(output_stem: Path) -> tuple[Path, Path, dict[str, int | float]]:
    event = read_event()
    start = parse_time(event["start_time"])
    lower, upper = start - timedelta(hours=24), start
    seconds, flux, flags = read_xrs_window(lower, upper)
    valid = np.isfinite(flux) & (flux > 0) & (flags == 0)
    intervals = read_catalog_intervals(lower, upper)
    other_flare = np.zeros(len(seconds), dtype=bool)
    for flare_start, flare_end, _ in intervals:
        other_flare |= (seconds >= int(flare_start.timestamp())) & (seconds <= int(flare_end.timestamp()))
    candidates = valid & ~other_flare
    threshold = float(event["start_threshold_flux_W_m2"])
    capped = candidates & (flux <= threshold)
    uncapped_background = float(event["background_uncapped_W_m2"])
    counts: dict[str, int | float] = {
        "observations": len(seconds),
        "valid_quality": int(valid.sum()),
        "other_flare_masked": int((valid & other_flare).sum()),
        "uncapped_candidates": int(candidates.sum()),
        "capped_retained": int(capped.sum()),
        "start_threshold_W_m2": threshold,
        "uncapped_background_W_m2": uncapped_background,
    }

    if counts["observations"] != 1440 or counts["valid_quality"] != 1440:
        raise ValueError(f"Unexpected source window coverage: {counts}")
    if counts["uncapped_candidates"] != int(event["uncapped_retained_minutes"]):
        raise ValueError("Recreated un-capped candidate count differs from the saved event calculation")
    if counts["capped_retained"] != int(event["capped_retained_minutes"]):
        raise ValueError("Recreated capped count differs from the saved event calculation")
    calculated_mean = float(np.mean(flux[candidates]))
    if not math.isclose(calculated_mean, uncapped_background, rel_tol=2e-6):
        raise ValueError("Recreated un-capped mean differs from the saved event calculation")

    times = np.asarray([datetime.fromtimestamp(int(value), UTC) for value in seconds], dtype=object)
    fig, (ax, band_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.55], "hspace": 0.08},
    )
    ax.plot(times[valid], flux[valid], color="#9aa6b2", linewidth=0.75, alpha=0.8,
            label="Valid 1-minute XRS-B observations", zorder=1)
    ax.scatter(times[candidates], flux[candidates], color="#1769aa", s=7, alpha=0.75,
               label="After other-flare mask (uncapped candidates)", zorder=2)
    ax.axhline(threshold, color="#c62828", linewidth=1.8, linestyle="--",
               label="Start-time flux cap")
    ax.axhline(uncapped_background, color="#00897b", linewidth=1.5, linestyle=":",
               label="Uncapped 24-hour mean")
    ax.scatter([start], [threshold], marker="D", s=42, color="#c62828",
               edgecolor="white", linewidth=0.7, zorder=5,
               label="Exact start-time measurement (cap value)")
    ax.axvline(start, color="#c62828", linewidth=0.8, alpha=0.6)
    ax.set_yscale("log")
    ymin = max(float(np.nanmin(flux[valid])) * 0.65, 1e-10)
    ymax = float(np.nanmax(flux[valid])) * 1.5
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(lower, start + timedelta(minutes=25))
    ax.set_ylabel("GOES XRS-B irradiance (W m$^{-2}$)")
    ax.grid(True, which="both", color="#d7dee5", linewidth=0.55, alpha=0.75)
    fig.legend(loc="upper center", bbox_to_anchor=(0.53, 0.99), fontsize=8, ncol=3,
               frameon=False, columnspacing=1.2, handlelength=2.2)
    ax.text(
        0.99, 0.97,
        f"Candidates after quality + flare filters: {int(candidates.sum()):,}\n"
        f"At or below cap: {int(capped.sum()):,}  →  capped background unavailable",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.94},
    )

    for flare_start, flare_end, _ in intervals:
        left, right = max(flare_start, lower), min(flare_end, upper)
        band_ax.axvspan(left, right, color="#e6a23c", alpha=0.8, linewidth=0)
    band_ax.set_ylim(0, 1)
    band_ax.set_yticks([])
    band_ax.set_ylabel("Other\nflares", rotation=0, ha="right", va="center", labelpad=33, fontsize=8)
    band_ax.set_xlabel("UTC time in the 24 hours before flare start")
    band_ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=UTC))
    band_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=UTC))
    band_ax.grid(True, axis="x", color="#d7dee5", linewidth=0.55, alpha=0.75)
    band_ax.legend(handles=[Patch(facecolor="#e6a23c", alpha=0.8, label=f"Other catalog flare intervals ({len(intervals)})")],
                   loc="upper left", bbox_to_anchor=(0, -0.35), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    png, svg, counts = make_figure(args.output_stem.resolve())
    print(f"PNG: {png}\nSVG: {svg}\nCounts: {counts}")


if __name__ == "__main__":
    main()
