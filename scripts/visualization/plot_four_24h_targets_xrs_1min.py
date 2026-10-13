#!/usr/bin/env python3
"""Render one-minute XRS-B line plots for four 24-hour targets.

Uses actual NOAA/NCEI GOES XRS L2 ``avg1m`` XRS-B observations and catalogued
NOAA flare-summary events. Maximum figures highlight one event; cumulative
figures mark every contributing event. RXFI panels show background-to-peak rise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


WINDOW_START = datetime(2024, 6, 16, 1, 11, tzinfo=timezone.utc)
WINDOW_END = WINDOW_START + timedelta(hours=24)


def catalog_events(path: Path) -> list[dict[str, object]]:
    events = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                time = datetime.fromisoformat(row["time"]).replace(tzinfo=timezone.utc)
                peak, background = float(row["xrsb_irrad"]), float(row["background_irrad"])
            except (KeyError, ValueError, TypeError):
                continue
            if WINDOW_START <= time < WINDOW_END and peak > 0 and background > 0:
                events.append({"time": time, "hours": (time - WINDOW_START).total_seconds() / 3600, "peak": peak, "background": background, "rxfi": (peak - background) / background, "flare_id": row["flare_id"]})
    if not events:
        raise ValueError("No valid flare-summary events in the selected window")
    return sorted(events, key=lambda event: float(event["hours"]))


def one_minute_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as dataset:
        times = num2date(dataset["time"][:], dataset["time"].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        seconds = np.asarray([value.replace(tzinfo=timezone.utc).timestamp() for value in times], dtype=float)
        flux = np.asarray(dataset["xrsb_flux"][:], dtype=float)
        flags = np.asarray(dataset["xrsb_flag"][:])
    lower, upper = WINDOW_START.timestamp(), WINDOW_END.timestamp()
    valid = (seconds >= lower) & (seconds < upper) & np.isfinite(flux) & (flux > 0) & (flags == 0)
    if not valid.any():
        raise ValueError("No valid one-minute XRS-B observations in the selected window")
    return (seconds[valid] - lower) / 3600, flux[valid]


def line_value(hours: np.ndarray, flux: np.ndarray, event_hour: float) -> float:
    return float(np.interp(event_hour, hours, flux))


def base_plot(hours: np.ndarray, flux: np.ndarray):
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.plot(hours, flux, color="#173f5f", linewidth=1.05, solid_capstyle="round", zorder=1)
    axis.set_xlim(0, 24)
    axis.set_ylim(max(float(np.nanmin(flux)) * 0.82, 1e-9), float(np.nanmax(flux)) * 1.16)
    tick_hours = (0, 6, 12, 18, 24)
    axis.set_xticks(tick_hours)
    axis.set_xticklabels([(WINDOW_START + timedelta(hours=hour)).strftime("%Y-%m-%d\n%H:%M") for hour in tick_hours])
    axis.set_xlabel("UTC time", color="#24445d", fontsize=10.5)
    axis.set_ylabel("X-ray flux (W m$^{-2}$)", color="#24445d", fontsize=10.5)
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9fb3c8"); axis.spines["bottom"].set_color("#9fb3c8")
    axis.tick_params(colors="#526a7a", labelsize=9)
    axis.grid(axis="y", color="#e7eef4", linewidth=0.8, zorder=0)
    return figure, axis


def mark_peak(axis, event: dict[str, object], hours: np.ndarray, flux: np.ndarray, color: str) -> None:
    x, y = float(event["hours"]), float(event["peak"])
    axis.vlines(x, axis.get_ylim()[0], y, color=color, linewidth=1.25, alpha=0.9, zorder=3)
    axis.scatter([x], [y], s=41, color=color, edgecolor="white", linewidth=0.9, zorder=4)


def mark_rxfi(axis, event: dict[str, object], hours: np.ndarray, flux: np.ndarray, color: str, emphasize: bool = False) -> None:
    x, peak_on_line = float(event["hours"]), float(event["peak"])
    background = float(event["background"])
    linewidth, marker_scale = (3.0, 1.4) if emphasize else (1.65, 1.0)
    axis.vlines(x, background, peak_on_line, color=color, linewidth=linewidth, alpha=0.92, zorder=3)
    axis.hlines([background, peak_on_line], x - 0.13, x + 0.13, color=color, linewidth=linewidth * 0.70, zorder=3)
    axis.scatter([x, x], [background, peak_on_line], s=[20 * marker_scale, 43 * marker_scale], color=color, edgecolor="white", linewidth=0.8, zorder=4)
    if emphasize:
        axis.annotate("largest relative increase", xy=(x, (background + peak_on_line) / 2), xytext=(0.98, 0.84), textcoords="axes fraction", ha="right", color="#513ea1", fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "-", "color": "#7158d6", "lw": 1.2})


def save_plot(path: Path, hours: np.ndarray, flux: np.ndarray, events: list[dict[str, object]], kind: str) -> None:
    figure, axis = base_plot(hours, flux)
    if kind == "max_peak_flux":
        maximum = max(events, key=lambda event: float(event["peak"]))
        mark_peak(axis, maximum, hours, flux, "#f36f45")
        axis.annotate("maximum peak", xy=(float(maximum["hours"]), float(maximum["peak"])), xytext=(0.98, 0.84), textcoords="axes fraction", ha="right", color="#b64625", fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "-", "color": "#f36f45", "lw": 1.2})
    elif kind == "cumulative_peak_flux":
        for event in events: mark_peak(axis, event, hours, flux, "#f36f45")
    elif kind == "max_rxfi":
        mark_rxfi(axis, max(events, key=lambda event: float(event["rxfi"])), hours, flux, "#7158d6", emphasize=True)
    elif kind == "cumulative_rxfi":
        for event in events: mark_rxfi(axis, event, hours, flux, "#7158d6")
    else:
        raise ValueError(kind)
    figure.savefig(path, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flare-catalog", type=Path, required=True)
    parser.add_argument("--xrs-avg1m", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/noaa_flare_target_figure/one_minute_xrs"))
    args = parser.parse_args()
    events, (hours, flux) = catalog_events(args.flare_catalog), one_minute_series(args.xrs_avg1m)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi"):
        save_plot(args.output_dir / f"{kind}_one_minute_xrs.svg", hours, flux, events, kind)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "window": f"[{WINDOW_START.isoformat()}, {WINDOW_END.isoformat()})", "event_count": len(events), "one_minute_valid_sample_count": int(len(flux)),
        "flare_catalog": str(args.flare_catalog), "flare_catalog_sha256": digest(args.flare_catalog),
        "xrs_avg1m": str(args.xrs_avg1m), "xrs_avg1m_sha256": digest(args.xrs_avg1m),
        "xrs_field": "xrsb_flux", "valid_rule": "finite, positive, xrsb_flag == 0", "satellite": "GOES-18",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
