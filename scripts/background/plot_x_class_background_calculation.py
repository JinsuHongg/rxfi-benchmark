#!/usr/bin/env python3
"""Visualize the project's lower-50% 24-hour background for an X-class flare."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_background.calculation import mask_other_flare_samples, valid_measurement_mask
from paper_background.noaa import read_l2_year, verify_input_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/background_lower50.json"
INPUT_LOCK = ROOT / "configs/paper_background_sensitivity.inputs.lock.json"
CATALOG_DIRECTORY = ROOT / "data/raw/noaa_paper_background"
XRS_DIRECTORY = CATALOG_DIRECTORY / "xrs_l2"
RUN_FINGERPRINT = "a7e506fa45a732128ad3192cc023661000c6dab37f9633c320b21dab4ff1cb55"
DEFAULT_EVENT_RESULTS = ROOT / "outputs/background_lower50/runs" / RUN_FINGERPRINT / "event_backgrounds.csv"
DEFAULT_OUTPUT_STEM = ROOT / "outputs/background_lower50/figures/x2_1_20110922_background_calculation"
DEFAULT_EVENT_ID = "201109221025"
UTC = timezone.utc

COLORS = {
    "trace": "#536878",
    "selected": "#007f73",
    "background": "#007f73",
    "masked_flare": "#d97706",
    "flare": "#7c3aed",
    "window": "#2563eb",
    "grid": "#d8dee6",
}


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_pinned_inputs() -> tuple[dict, dict, Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    catalog_lock = lock["catalog"]
    catalog_path = CATALOG_DIRECTORY / catalog_lock["filename"]
    verify_input_file(catalog_path, catalog_lock["sha256"], catalog_lock["size_bytes"])
    xrs_lock = next(
        item for item in lock["xrs_l2"]["files"]
        if item["satellite"] == 15 and item["year"] == 2011
    )
    xrs_path = XRS_DIRECTORY / "g15" / "2011.nc"
    verify_input_file(xrs_path, xrs_lock["sha256"], xrs_lock["size_bytes"])
    return config, lock, catalog_path


def _event_row(path: Path, event_id: str) -> dict[str, str]:
    matches = [row for row in _read_csv(path) if row.get("flare_id") == event_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one event {event_id} in {path}; found {len(matches)}")
    event = matches[0]
    if not event.get("flare_class", "").startswith("X"):
        raise ValueError(f"Event {event_id} is not X class: {event.get('flare_class')}")
    return event


def _catalog_intervals(
    catalog_path: Path,
    left: datetime,
    right: datetime,
    target_event_id: str,
) -> list[tuple[int, int, str]]:
    intervals: list[tuple[int, int, str]] = []
    left_s, right_s = int(left.timestamp()), int(right.timestamp())
    for row in _read_csv(catalog_path):
        if row.get("flare_id", "").strip() == target_event_id:
            continue
        if not row.get("start_time") or not row.get("end_time"):
            continue
        start, end = parse_time(row["start_time"]), parse_time(row["end_time"])
        start_s, end_s = int(start.timestamp()), int(end.timestamp())
        if end_s >= left_s and start_s < right_s:
            intervals.append((start_s, end_s, row.get("flare_id", "")))
    return intervals


def _verify_calculation(event: dict[str, str], observed: dict[str, float | int]) -> None:
    checks = {
        "observed_minutes": observed["observed_minutes"],
        "valid_quality_minutes": observed["valid_quality_minutes"],
        "other_flare_masked_minutes": observed["other_flare_masked_minutes"],
        "uncapped_retained_minutes": observed["eligible_minutes"],
        "lower_retained_minutes": observed["selected_minutes"],
    }
    for field, calculated in checks.items():
        expected = int(event[field])
        if expected != int(calculated):
            raise ValueError(f"{field} mismatch: event table={expected}, recreated={calculated}")
    for field, calculated in (
        ("background_lower_fraction_W_m2", observed["background"]),
        ("rxfi_lower_fractional_excess", observed["rxfi"]),
    ):
        expected = float(event[field])
        if not math.isclose(expected, float(calculated), rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(f"{field} mismatch: event table={expected:.12g}, recreated={calculated:.12g}")


def _load_event_data(event: dict[str, str], catalog_path: Path) -> dict:
    start = parse_time(event["start_time"])
    end = parse_time(event["end_time"])
    peak_time = parse_time(event["time"])
    left = start - timedelta(hours=24)
    seconds, flux, flags, _ = read_l2_year(XRS_DIRECTORY / "g15" / "2011.nc")
    plot_right_s = int((end + timedelta(minutes=16)).timestamp())
    plot_left_s = int(left.timestamp())
    view = (seconds >= plot_left_s) & (seconds <= plot_right_s)
    seconds, flux, flags = seconds[view], flux[view], flags[view]

    in_window = (seconds >= plot_left_s) & (seconds < int(start.timestamp()))
    valid = valid_measurement_mask(flux, flags)
    intervals = _catalog_intervals(catalog_path, left, start, event["flare_id"])
    other_flare = mask_other_flare_samples(seconds, intervals, event["flare_id"])
    eligible = in_window & valid & ~other_flare
    eligible_indices = np.flatnonzero(eligible)
    n_select = math.ceil(len(eligible_indices) * float(event["lower_fraction"]))
    if not n_select:
        raise ValueError("No eligible measurements remain for the lower-half background")
    sorted_order = np.argsort(flux[eligible_indices], kind="stable")
    selected_indices = eligible_indices[sorted_order[:n_select]]
    selected = np.zeros(len(seconds), dtype=bool)
    selected[selected_indices] = True
    background = float(np.mean(flux[selected]))
    peak = float(event["peak_flux_W_m2"])
    rxfi = (peak - background) / background
    observed = {
        "observed_minutes": int(in_window.sum()),
        "valid_quality_minutes": int((in_window & valid).sum()),
        "quality_rejected_minutes": int((in_window & ~valid).sum()),
        "other_flare_masked_minutes": int((in_window & valid & other_flare).sum()),
        "eligible_minutes": int(eligible.sum()),
        "selected_minutes": int(selected.sum()),
        "background": background,
        "rxfi": rxfi,
    }
    _verify_calculation(event, observed)

    times = np.asarray([datetime.fromtimestamp(int(sec), UTC) for sec in seconds], dtype=object)
    return {
        "start": start,
        "end": end,
        "peak_time": peak_time,
        "left": left,
        "seconds": seconds,
        "times": times,
        "flux": flux,
        "flags": flags,
        "valid": valid,
        "in_window": in_window,
        "other_flare": other_flare,
        "eligible": eligible,
        "selected": selected,
        "eligible_values": flux[eligible],
        "background": background,
        "observed": observed,
        "intervals": intervals,
    }


def make_plot(event_id: str, event_results: Path, output_stem: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10.5,
    })
    config, lock, catalog_path = _read_pinned_inputs()
    event = _event_row(event_results, event_id)
    if event["satellite"] != "GOES-15" or parse_time(event["start_time"]).year != 2011:
        raise ValueError("This figure is pinned to the 2011 GOES-15 event")
    data = _load_event_data(event, catalog_path)
    observed = data["observed"]
    lower_fraction = float(config["method"]["lower_fraction"])
    peak_flux = float(event["peak_flux_W_m2"])

    fig, ax_time = plt.subplots(figsize=(13.2, 6.2))
    left, start, end = data["left"], data["start"], data["end"]
    times, flux = data["times"], data["flux"]
    valid, selected = data["valid"], data["selected"]

    ax_time.axvspan(left, start, color=COLORS["window"], alpha=0.055,
                    linewidth=0, zorder=0)
    for interval_start, interval_end, _ in data["intervals"]:
        lo = max(datetime.fromtimestamp(interval_start, UTC), left)
        hi = min(datetime.fromtimestamp(interval_end, UTC), start)
        if hi >= lo:
            ax_time.axvspan(lo, hi, color=COLORS["masked_flare"], alpha=0.15,
                            linewidth=0, zorder=1)
    ax_time.axvspan(start, end, color=COLORS["flare"], alpha=0.11, linewidth=0, zorder=0)
    ax_time.axvline(left, color=COLORS["window"], linestyle=":", linewidth=1.0, zorder=2)
    ax_time.plot(times[valid], flux[valid], color=COLORS["trace"], linewidth=0.85,
                 alpha=0.76, zorder=2, label="X-ray flux")
    excluded_flare_samples = data["in_window"] & valid & data["other_flare"]
    ax_time.scatter(times[excluded_flare_samples], flux[excluded_flare_samples],
                    s=14, color=COLORS["masked_flare"], alpha=0.9,
                    edgecolors="white", linewidths=0.25, zorder=4,
                    label="Other flare intervals excluded")
    ax_time.scatter(times[selected], flux[selected], s=9, color=COLORS["selected"],
                    alpha=0.78, linewidths=0, zorder=3,
                    label=f"Selected lowest {lower_fraction:.0%} samples")

    background = data["background"]
    ax_time.hlines(background, left, start, color=COLORS["background"],
                   linestyle="--", linewidth=2.0, zorder=4,
                   label=f"Lower-{lower_fraction:.0%} mean background ($B_{{50}}$)")
    ax_time.axvline(start, color=COLORS["flare"], linestyle="--", linewidth=1.0,
                    alpha=0.85, zorder=4)
    ax_time.annotate(
        "", xy=(start, 0.955), xytext=(left, 0.955),
        xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "|-|", "color": COLORS["window"], "linewidth": 1.2},
        annotation_clip=False,
    )
    ax_time.text(
        0.45, 0.91,
        "24-hour window",
        transform=ax_time.transAxes, color=COLORS["window"], fontsize=15,
        ha="center", va="top",
    )
    ax_time.scatter([data["peak_time"]], [peak_flux], marker="*", s=150,
                    color=COLORS["flare"], edgecolor="white", linewidth=0.7, zorder=5)
    positive = flux[valid]
    background_label_time = left + timedelta(hours=1)
    ax_time.text(background_label_time, background * 1.14,
                 f"$B_{{50}}$ = {background:.3e} W m$^{{-2}}$",
                 color=COLORS["background"], fontsize=11, ha="left", va="bottom")
    ax_time.set_yscale("log")
    right = datetime.fromtimestamp(int(data["seconds"][-1]), UTC)
    ax_time.set_xlim(left, right)
    ax_time.set_ylim(max(float(np.min(positive)) * 0.65, 1e-10), float(np.max(positive)) * 1.45)
    ax_time.set_ylabel("X-ray flux (W m$^{-2}$)", fontsize=14)
    ax_time.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=UTC))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=UTC))
    ax_time.tick_params(axis="both", labelsize=12)
    ax_time.grid(True, which="both", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    ax_time.set_axisbelow(True)
    ax_time.set_xlabel("UTC time")
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS["trace"], linewidth=1.0, label="X-ray flux"),
            Line2D([], [], marker="o", linestyle="", color=COLORS["selected"], markersize=5,
                   label=f"Lowest {lower_fraction:.0%} samples used (n={observed['selected_minutes']:,})"),
            Line2D([], [], color=COLORS["background"], linestyle="--", linewidth=2,
                   label="$B_{50}$ background"),
            Line2D([], [], marker="o", linestyle="", color=COLORS["masked_flare"], markersize=5,
                   label="Other flare intervals excluded"),
            Patch(facecolor=COLORS["flare"], alpha=0.16,
                  label=f"{event['flare_class']} event interval"),
            Line2D([], [], marker="*", linestyle="", color=COLORS["flare"], markersize=9,
                   label="Event peak"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3,
        frameon=False, fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Event: {event['flare_class']} ({event_id}), {event['satellite']}")
    print(f"Counts: {observed['observed_minutes']} observed; {observed['valid_quality_minutes']} quality-valid; "
          f"{observed['other_flare_masked_minutes']} other-flare masked; "
          f"{observed['eligible_minutes']} eligible; {observed['selected_minutes']} selected")
    print(f"B50: {background:.12g} W m^-2")
    print(f"PNG: {png}\nSVG: {svg}")
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID,
                        help="NOAA flare ID (default: the 2011-09-22 X2.1 event)")
    parser.add_argument("--event-results", type=Path, default=DEFAULT_EVENT_RESULTS,
                        help="lower-50% event_backgrounds.csv from the pinned calculation run")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM,
                        help="output filename stem; PNG and SVG extensions are added")
    args = parser.parse_args()
    make_plot(args.event_id, args.event_results.resolve(), args.output_stem.resolve())


if __name__ == "__main__":
    main()
