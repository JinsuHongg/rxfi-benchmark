#!/usr/bin/env python3
"""Plot the three X-class flares with unavailable capped backgrounds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/raw/noaa_paper_background/sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv"
EVENT_RESULTS = ROOT / "outputs/paper_background_sensitivity/event_backgrounds.csv"
XRS_DIRECTORY = ROOT / "data/raw/noaa_paper_background/xrs_l2"
DEFAULT_OUTPUT = ROOT / "outputs/paper_background_sensitivity/figures/missing_x_class_backgrounds"
UTC = timezone.utc


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def read_events() -> list[dict[str, str]]:
    with EVENT_RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = [
        row for row in rows
        if row["capped_status"] != "ok" and row["flare_class"].startswith("X")
    ]
    if len(events) != 3:
        raise ValueError(f"Expected the three missing-background X-class events; found {len(events)}")
    return events


def catalog_intervals(left: datetime, right: datetime, event_id: str) -> list[tuple[datetime, datetime]]:
    intervals = []
    with CATALOG.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("flare_id", "").strip() == event_id:
                continue
            if not row.get("start_time") or not row.get("end_time"):
                continue
            start, end = parse_time(row["start_time"]), parse_time(row["end_time"])
            if end >= left and start <= right:
                intervals.append((start, end))
    return intervals


def xrs_path(event: dict[str, str]) -> Path:
    satellite = "g" + event["satellite"].split("-")[-1]
    year = parse_time(event["start_time"]).year
    path = XRS_DIRECTORY / satellite / f"{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing local XRS Level 2 source: {path}")
    return path


def read_xrs(path: Path, left: datetime, right: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        time_var = ds.variables["time"]
        dates = num2date(
            time_var[:], time_var.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        seconds = np.asarray([int(d.replace(tzinfo=UTC).timestamp()) for d in dates], dtype=np.int64)
        flux = np.asarray(ds.variables["xrsb_flux"][:], dtype=float)
        flags = np.asarray(ds.variables["xrsb_flag"][:])
    lo = int(left.timestamp())
    hi = int(right.timestamp())
    start = int(np.searchsorted(seconds, lo, side="left"))
    end = int(np.searchsorted(seconds, hi, side="right"))
    return seconds[start:end], flux[start:end], flags[start:end]


def make_plot(output_stem: Path) -> tuple[Path, Path]:
    events = sorted(read_events(), key=lambda row: parse_time(row["start_time"]))
    fig, axes = plt.subplots(len(events), 1, figsize=(13, 10), sharex=False)

    for ax, event in zip(axes, events):
        start = parse_time(event["start_time"])
        end = parse_time(event["end_time"])
        peak_time = parse_time(event["time"])
        left = start - timedelta(hours=24)
        right = end + timedelta(minutes=45)
        seconds, flux, flags = read_xrs(xrs_path(event), left, right)
        times = np.asarray([datetime.fromtimestamp(int(sec), UTC) for sec in seconds], dtype=object)
        valid = np.isfinite(flux) & (flux > 0) & (flags == 0)
        in_prior = seconds < int(start.timestamp())
        intervals = catalog_intervals(left, start, event["flare_id"])
        in_other_flare = np.zeros(len(seconds), dtype=bool)
        for flare_start, flare_end in intervals:
            in_other_flare |= (
                (seconds >= int(flare_start.timestamp()))
                & (seconds <= int(flare_end.timestamp()))
            )
            ax.axvspan(max(flare_start, left), min(flare_end, start),
                       color="#e6a23c", alpha=0.16, linewidth=0, zorder=0)

        candidates = valid & in_prior & ~in_other_flare
        threshold = float(event["start_threshold_flux_W_m2"])
        capped = candidates & (flux <= threshold)
        if int(candidates.sum()) != int(event["uncapped_retained_minutes"]):
            raise ValueError(f"Recreated candidate count differs for {event['flare_id']}")
        if int(capped.sum()) != int(event["capped_retained_minutes"]):
            raise ValueError(f"Recreated capped count differs for {event['flare_id']}")
        ax.scatter(times[valid & ~candidates], flux[valid & ~candidates],
                   color="#aab4bf", s=5, alpha=0.45, linewidths=0, zorder=1)
        ax.scatter(times[candidates], flux[candidates],
                   color="#1769aa", s=7, alpha=0.72, linewidths=0, zorder=2)
        after_start = valid & (seconds >= int(start.timestamp()))
        ax.plot(times[after_start], flux[after_start], color="#7b4ab5", linewidth=1.0,
                alpha=0.8, zorder=3)
        ax.axhline(threshold, color="#c62828", linestyle="--", linewidth=1.5, zorder=2)
        ax.axvline(start, color="#c62828", linestyle=":", linewidth=1.0, zorder=2)
        ax.axvline(peak_time, color="#5e35b1", linestyle="-.", linewidth=0.9, alpha=0.8, zorder=2)

        # Mark the measured one-minute sample at the catalog start and peak times when present.
        for mark_time, marker, color in ((start, "D", "#c62828"), (peak_time, "*", "#5e35b1")):
            idx = int(np.searchsorted(seconds, int(mark_time.timestamp())))
            if idx < len(seconds) and seconds[idx] == int(mark_time.timestamp()) and valid[idx]:
                ax.scatter([times[idx]], [flux[idx]], marker=marker, color=color,
                           edgecolor="white", linewidth=0.5, s=60 if marker == "*" else 34, zorder=5)

        class_name = event["flare_class"]
        ax.set_title(
            f"{class_name}  |  {event['flare_id']}  |  {event['satellite']}  |  "
            f"capped samples: {int(capped.sum())}/{int(candidates.sum())}",
            loc="left", fontsize=10, fontweight="bold", pad=5,
        )
        ax.text(
            0.995, 0.93,
            f"Start-time cap: {threshold:.2e} W m$^{{-2}}$\n"
            f"Valid prior candidates: {int(candidates.sum()):,}; at/below cap: {int(capped.sum()):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.92},
        )
        ax.set_yscale("log")
        ax.set_xlim(left, right)
        finite = flux[valid]
        ax.set_ylim(max(float(np.min(finite)) * 0.6, 1e-10), float(np.max(finite)) * 1.7)
        ax.set_ylabel("XRS-B flux\n(W m$^{-2}$)", fontsize=8)
        ax.grid(True, which="both", color="#d7dee5", linewidth=0.5, alpha=0.75)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=UTC))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=UTC))
        ax.tick_params(axis="x", labelsize=8)

    legend = [
        Line2D([], [], marker="o", linestyle="", color="#1769aa", markersize=4,
               label="Eligible prior 24 h samples after quality/other-flare filters"),
        Line2D([], [], color="#c62828", linestyle="--", label="Flux at flare start (upper cap)"),
        Line2D([], [], color="#7b4ab5", linewidth=1, label="XRS-B from flare start onward"),
        Line2D([], [], color="#aab4bf", marker="o", linestyle="", markersize=4,
               label="Other valid observations (masked or post-start)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=2, frameon=False, fontsize=8)
    axes[-1].set_xlabel("UTC (24 h before flare start through 45 min after catalog end)")
    fig.suptitle("X-class flares with unavailable capped background", y=1.01, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=190, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    png, svg = make_plot(args.output_stem.resolve())
    print(f"PNG: {png}\nSVG: {svg}")


if __name__ == "__main__":
    main()
