#!/usr/bin/env python3
"""Plot the 2011-09-22 GOES-15 XRS Level 2 X-class flare."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/xrs_lv2/sci_gxrs-l2-irrad_g15_d20110922_v0-1-0.nc"
OUTDIR = ROOT / "outputs/noaa_single_flare_classification"
WINDOW_START = np.datetime64("2011-09-22T07:00:00")
WINDOW_END = np.datetime64("2011-09-22T15:00:00")


def load_series() -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(INPUT, engine="h5netcdf") as ds:
        times = ds["time"].values
        flux = np.asarray(ds["b_flux"].values, dtype=float)
        flags = np.asarray(ds["b_flags"].values)
    keep = (
        (times >= WINDOW_START)
        & (times < WINDOW_END)
        & np.isfinite(flux)
        & (flux > 0)
        & (flags == 0)
    )
    valid_times = times[keep]
    valid_flux = flux[keep]
    # NOAA class boundaries are defined from the peak of 1-minute XRS-B flux.
    minute_times = valid_times.astype("datetime64[m]")
    unique_minutes, inverse = np.unique(minute_times, return_inverse=True)
    counts = np.bincount(inverse)
    minute_flux = np.bincount(inverse, weights=valid_flux) / counts
    return unique_minutes, minute_flux


def main() -> None:
    times, flux = load_series()
    if len(flux) == 0:
        raise RuntimeError(f"No valid XRS-B samples found in {INPUT}")
    peak_i = int(np.argmax(flux))
    peak_time = times[peak_i]
    peak_flux = float(flux[peak_i])
    if peak_flux < 1e-4:
        raise RuntimeError(f"Expected an X-class peak; observed {peak_flux:g} W m^-2")
    x_index = int(peak_flux / 1e-4 * 10) / 10

    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.105, right=0.81, top=0.82, bottom=0.17)

    # Flux-class intervals and their boundary values.
    ax.axhspan(1e-6, 1e-5, color="#e9f1f8", zorder=0)
    ax.axhspan(1e-5, 1e-4, color="#fff0dc", zorder=0)
    ax.axhspan(1e-4, 4e-4, color="#f9e7e8", zorder=0)
    thresholds = [
        (1e-8, "A1", "#7d8994"),
        (1e-7, "B1", "#a9b0b8"),
        (1e-6, "C1", "#6585a1"),
        (1e-5, "M1", "#bd7620"),
        (1e-4, "X1", "#aa535c"),
    ]
    for value, label, color in thresholds:
        ax.axhline(value, color=color, linewidth=1.8, linestyle=(0, (5, 5)), zorder=1)
        ax.text(
            1.012, value, f"{label}  {value:.0e}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=18, color=color, clip_on=False,
        )

    ax.plot(times, flux, color="#245b83", linewidth=2.7, solid_capstyle="round", zorder=3)
    ax.scatter([peak_time], [peak_flux], s=175, color="#d77a17", edgecolor="white",
               linewidth=2.2, zorder=5)
    peak_label = f"{peak_flux:.2e} W m⁻²\nX{x_index:.1f}  ·  11:01 UTC"
    ax.annotate(
        peak_label,
        xy=(peak_time, peak_flux), xycoords="data",
        xytext=(55, 40), textcoords="offset points",
        fontsize=21, fontweight="bold", color="#282c31",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#d77a17", "linewidth": 1.8},
        arrowprops={"arrowstyle": "-", "color": "#d77a17", "linewidth": 1.8},
        zorder=6,
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-9, 5e-4)
    ax.set_xlim(WINDOW_START, WINDOW_END)
    ax.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    ax.set_yticklabels([r"$10^{-9}$", r"$10^{-8}$", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$"])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[7, 9, 11, 13, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="y", which="major", color="#d9dee3", linewidth=1.0)
    ax.grid(axis="x", which="major", color="#edf0f2", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=10, colors="#353b41")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#87919a")
    ax.spines["bottom"].set_color("#87919a")
    ax.set_xlabel("Time (UTC) · 22 September 2011", labelpad=18, fontweight="semibold", color="#353b41")
    ax.set_ylabel(r"X-ray flux (W m$^{-2}$)", labelpad=18, fontweight="semibold", color="#353b41")

    # Intentionally no title, subtitle, or figure caption for slide placement.
    png = OUTDIR / "goes15_20110922_single_flare.png"
    svg = OUTDIR / "goes15_20110922_single_flare.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    print(f"valid_minutes={len(flux)}; peak={peak_flux:.12g} W m^-2; class=X{x_index:.1f}; peak_time={peak_time}")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
