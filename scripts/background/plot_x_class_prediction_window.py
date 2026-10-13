#!/usr/bin/env python3
"""Plot NOAA flare peaks in the 24-hour window leading to an X-class flare."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from paper_background.calculation import valid_measurement_mask
from paper_background.noaa import read_l2_year, verify_input_file


ROOT = Path(__file__).resolve().parents[2]
INPUT_LOCK = ROOT / "configs/paper_background_sensitivity.inputs.lock.json"
CATALOG_DIRECTORY = ROOT / "data/raw/noaa_paper_background"
XRS_FILE = CATALOG_DIRECTORY / "xrs_l2/g15/2011.nc"
OUTPUT_STEM = ROOT / "outputs/background_lower50/figures/x2_1_20110922_prediction_window"
UTC = timezone.utc
WINDOW_START = datetime(2011, 9, 21, 12, tzinfo=UTC)
WINDOW_END = datetime(2011, 9, 22, 12, tzinfo=UTC)
X_CLASS_EVENT_ID = "201109221025"


def read_catalog() -> tuple[list[dict[str, str]], dict[str, str]]:
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    catalog_lock = lock["catalog"]
    catalog_path = CATALOG_DIRECTORY / catalog_lock["filename"]
    verify_input_file(catalog_path, catalog_lock["sha256"], catalog_lock["size_bytes"])
    xrs_lock = next(
        item for item in lock["xrs_l2"]["files"]
        if item["satellite"] == 15 and item["year"] == 2011
    )
    verify_input_file(XRS_FILE, xrs_lock["sha256"], xrs_lock["size_bytes"])
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = [
        row for row in rows
        if WINDOW_START <= datetime.fromisoformat(row["time"]).replace(tzinfo=UTC) < WINDOW_END
    ]
    if not events:
        raise ValueError("No NOAA flare catalog events found in the 24-hour prediction window")
    if any(row["xrsb_irrad_source"] != "GOES-15" for row in events):
        raise ValueError("The selected events do not all use GOES-15 XRS-B peak measurements")
    if not any(row["flare_id"] == X_CLASS_EVENT_ID and row["flare_class"].startswith("X") for row in events):
        raise ValueError(f"Expected X-class target event {X_CLASS_EVENT_ID} in the window")
    return events, lock["catalog"]


def make_plot(output_stem: Path = OUTPUT_STEM) -> tuple[Path, Path]:
    events, _ = read_catalog()
    seconds, flux, flags, _ = read_l2_year(XRS_FILE)
    start_s, end_s = int(WINDOW_START.timestamp()), int(WINDOW_END.timestamp())
    in_window = (seconds >= start_s) & (seconds <= end_s)
    seconds, flux, flags = seconds[in_window], flux[in_window], flags[in_window]
    valid = valid_measurement_mask(flux, flags)
    if not valid.any():
        raise ValueError("No valid GOES-15 XRS-B measurements found in the prediction window")

    times = np.asarray([datetime.fromtimestamp(int(value), UTC) for value in seconds], dtype=object)
    observed_flux = np.where(valid, flux, np.nan)
    event_times = [datetime.fromisoformat(row["time"]).replace(tzinfo=UTC) for row in events]
    event_flux = np.asarray([float(row["xrsb_irrad"]) for row in events], dtype=float)
    if not np.isfinite(event_flux).all() or np.any(event_flux <= 0):
        raise ValueError("NOAA catalog contains missing or non-positive event peak irradiance")

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.plot(times, observed_flux, color="#536878", linewidth=1.05, zorder=2)
    ax.scatter(
        event_times,
        event_flux,
        marker="*",
        s=190,
        color="#d62728",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    for row, event_time, peak_flux in zip(events, event_times, event_flux, strict=True):
        ax.annotate(
            row["flare_class"],
            (event_time, peak_flux),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#b31b1b",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=6,
        )

    ax.set_xlim(WINDOW_START, WINDOW_END)
    positive = flux[valid]
    ax.set_ylim(max(float(positive.min()) * 0.65, 1e-10), float(positive.max()) * 1.6)
    ax.set_yscale("log")
    ax.set_ylabel("X-ray flux (W m$^{-2}$)")
    ax.set_xlabel("UTC time")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=UTC))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=UTC))
    ax.grid(True, which="both", color="#d8dee6", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    classes = ", ".join(row["flare_class"] for row in events)
    print(f"Window: {WINDOW_START.isoformat()} to {WINDOW_END.isoformat()} (UTC)")
    print(f"NOAA flare peaks: {len(events)} ({classes})")
    print(f"GOES-15 valid XRS-B avg1m samples: {int(valid.sum())}/{len(valid)}")
    print(f"PNG: {png}\nSVG: {svg}")
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM,
                        help="output filename stem; PNG and SVG extensions are added")
    args = parser.parse_args()
    make_plot(args.output_stem.resolve())


if __name__ == "__main__":
    main()
