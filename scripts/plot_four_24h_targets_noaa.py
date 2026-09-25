#!/usr/bin/env python3
"""Create a data-backed four-target 24-hour NOAA flare-summary figure.

The figure uses event-level NOAA/NCEI XRS Flare Report fields ``xrsb_irrad``
and ``background_irrad``. It does not synthesize a continuous XRS time series:
each mark is a catalogued flare peak in one fixed half-open 24-hour window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT_START = "2024-06-16 01:11:00"


def valid_events(path: Path, start: datetime) -> list[dict[str, object]]:
    end = start + timedelta(hours=24)
    events: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                peak_time = datetime.fromisoformat(row["time"])
                peak, background = float(row["xrsb_irrad"]), float(row["background_irrad"])
            except (KeyError, TypeError, ValueError):
                continue
            if start <= peak_time < end and peak > 0 and background > 0:
                events.append({
                    "time": peak_time, "hours": (peak_time - start).total_seconds() / 3600,
                    "peak_flux_w_m2": peak, "background_flux_w_m2": background,
                    "rxfi": (peak - background) / background,
                    "flare_class": row["flare_class"], "flare_id": row["flare_id"],
                })
    if not events:
        raise ValueError("No valid peak/background events in the requested 24-hour interval")
    return sorted(events, key=lambda event: float(event["hours"]))


def plot_events(axis, hours, values, color, ylabel, title, subtitle, highlight, annotation):
    markerline, stemlines, baseline = axis.stem(hours, values, linefmt=color, markerfmt="o", basefmt=" ")
    plt.setp(stemlines, linewidth=1.55, alpha=0.78)
    plt.setp(markerline, markersize=4.2, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5)
    axis.scatter([hours[highlight]], [values[highlight]], s=76, color=color, edgecolor="#102a43", linewidth=1.1, zorder=5)
    axis.annotate(annotation, xy=(hours[highlight], values[highlight]), xytext=(0.98, 0.92), textcoords="axes fraction", ha="right", va="top", fontsize=10.5, fontweight="bold", color="#102a43", arrowprops={"arrowstyle": "-", "color": "#526a7a", "lw": 1.1})
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#102a43", pad=24)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, fontsize=10.5, color="#526a7a")
    axis.set_ylabel(ylabel, fontsize=10.5, color="#24445d")


def make_figure(events: list[dict[str, object]], output: Path) -> None:
    hours = [float(event["hours"]) for event in events]
    peaks = [float(event["peak_flux_w_m2"]) * 1e6 for event in events]
    rxfi = [float(event["rxfi"]) for event in events]
    cumulative_peaks = []
    cumulative_rxfi = []
    total_peak = total_rxfi = 0.0
    for peak, relative in zip(peaks, rxfi):
        total_peak += peak; total_rxfi += relative
        cumulative_peaks.append(total_peak); cumulative_rxfi.append(total_rxfi)
    max_peak_index = max(range(len(peaks)), key=peaks.__getitem__)
    max_rxfi_index = max(range(len(rxfi)), key=rxfi.__getitem__)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.6), sharex=True, constrained_layout=True)
    figure.set_facecolor("#f8fafc")
    orange, purple, navy = "#f36f45", "#7158d6", "#102a43"
    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#e7eef4", linewidth=0.9)
        axis.set_xlim(0, 24)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        axis.tick_params(labelsize=9.5, colors="#526a7a")
        axis.spines["left"].set_color("#b7c8d8"); axis.spines["bottom"].set_color("#b7c8d8")

    plot_events(axes[0, 0], hours, peaks, orange, "Peak XRS-B irradiance (µW m⁻²)", "Max Peak X-ray Flux", "Tallest catalogued flare peak in this 24 h window", max_peak_index, f"max = {peaks[max_peak_index]:.2f} µW m⁻²")
    axes[0, 0].axhline(peaks[max_peak_index], color=orange, alpha=0.30, linewidth=1.2, linestyle="--")

    axes[0, 1].step([0, *hours], [0, *cumulative_peaks], where="post", color=orange, linewidth=2.35)
    axes[0, 1].fill_between([0, *hours], [0, *cumulative_peaks], step="post", color=orange, alpha=0.13)
    axes[0, 1].scatter(hours, cumulative_peaks, color=orange, s=12, zorder=4)
    axes[0, 1].set_title("Cumulative Peak X-ray Flux", loc="left", fontsize=16, fontweight="bold", color=navy, pad=24)
    axes[0, 1].text(0, 1.02, "Running sum of all catalogued peak irradiances", transform=axes[0, 1].transAxes, fontsize=10.5, color="#526a7a")
    axes[0, 1].set_ylabel("Cumulative peak irradiance (µW m⁻²)", fontsize=10.5, color="#24445d")
    axes[0, 1].annotate(f"sum = {total_peak:.1f} µW m⁻²", xy=(hours[-1], cumulative_peaks[-1]), xytext=(0.98, 0.92), textcoords="axes fraction", ha="right", va="top", fontsize=10.5, fontweight="bold", color=navy, arrowprops={"arrowstyle": "-", "color": "#526a7a", "lw": 1.1})

    plot_events(axes[1, 0], hours, rxfi, purple, "Relative increase, RXFI", "Max RXFI", "Largest catalogued background-to-peak increase", max_rxfi_index, f"max = {rxfi[max_rxfi_index]:.2f}×")
    axes[1, 0].axhline(rxfi[max_rxfi_index], color=purple, alpha=0.30, linewidth=1.2, linestyle="--")

    axes[1, 1].step([0, *hours], [0, *cumulative_rxfi], where="post", color=purple, linewidth=2.35)
    axes[1, 1].fill_between([0, *hours], [0, *cumulative_rxfi], step="post", color=purple, alpha=0.13)
    axes[1, 1].scatter(hours, cumulative_rxfi, color=purple, s=12, zorder=4)
    axes[1, 1].set_title("Cumulative RXFI", loc="left", fontsize=16, fontweight="bold", color=navy, pad=24)
    axes[1, 1].text(0, 1.02, "Running sum of all event-level relative increases", transform=axes[1, 1].transAxes, fontsize=10.5, color="#526a7a")
    axes[1, 1].set_ylabel("Cumulative relative increase", fontsize=10.5, color="#24445d")
    axes[1, 1].annotate(f"sum = {total_rxfi:.1f}", xy=(hours[-1], cumulative_rxfi[-1]), xytext=(0.98, 0.92), textcoords="axes fraction", ha="right", va="top", fontsize=10.5, fontweight="bold", color=navy, arrowprops={"arrowstyle": "-", "color": "#526a7a", "lw": 1.1})

    for axis in axes[1, :]: axis.set_xlabel("Hours since 2024-06-16 01:11 UTC", fontsize=10.5, color="#24445d")
    figure.suptitle("Four 24-hour solar-flare forecasting targets from one NOAA flare-summary window", fontsize=20, fontweight="bold", color=navy)
    figure.text(0.5, 0.002, "Source: NOAA/NCEI science-quality GOES XRS Flare Report, 2024; 34 catalogued events with finite positive XRS-B peak and background irradiance. RXFI = (peak − background) / background.", ha="center", fontsize=8.7, color="#526a7a")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaa-csv", type=Path, required=True)
    parser.add_argument("--start", default=DEFAULT_START, help="24-hour window start, ISO-like UTC timestamp")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/noaa_flare_target_figure"))
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start)
    events = valid_events(args.noaa_csv, start)
    stem = f"four_24h_targets_noaa_{start:%Y%m%dT%H%M}"
    make_figure(events, args.output_dir / f"{stem}.svg")
    with (args.output_dir / f"{stem}_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0])); writer.writeheader(); writer.writerows(events)
    digest = hashlib.sha256(args.noaa_csv.read_bytes()).hexdigest()
    (args.output_dir / f"{stem}_metadata.json").write_text(json.dumps({"source_csv": str(args.noaa_csv), "source_sha256": digest, "window_start_utc": start.isoformat(), "window_end_utc": (start + timedelta(hours=24)).isoformat(), "event_count": len(events), "target_definitions": {"max_peak_flux": "maximum xrsb_irrad", "cumulative_peak_flux": "sum xrsb_irrad", "max_rxfi": "maximum (xrsb_irrad-background_irrad)/background_irrad", "cumulative_rxfi": "sum event-level RXFI"}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
