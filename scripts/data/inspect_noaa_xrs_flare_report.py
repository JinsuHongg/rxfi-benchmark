#!/usr/bin/env python3
"""Inspect a NOAA/NCEI XRS Flare Report CSV and calculate candidate RXFI values.

The default is a small, official yearly catalog.  Pass a local CSV path or an
official NOAA/NCEI CSV URL to inspect another release.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from pathlib import Path
from typing import Iterable, TextIO
from urllib.request import urlopen


DEFAULT_CSV_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/multi/l2/data/xrsf-l2-flrpt_science/csv/"
    "sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv"
)

RXFI_COLUMNS = (
    "time",
    "start_time",
    "end_time",
    "flare_id",
    "xrsb_irrad",
    "flare_class",
    "xrsb_irrad_source",
    "background_irrad",
    "peak_saturated",
    "sequential_flare_num",
)
REQUIRED_COLUMNS = {"xrsb_irrad", "background_irrad"}


def open_csv(source: str) -> TextIO:
    """Open a local path or HTTPS CSV URL as UTF-8 text."""
    if source.startswith(("https://", "http://")):
        response = urlopen(source, timeout=60)  # noqa: S310 - user selects source.
        return io.TextIOWrapper(response, encoding="utf-8", newline="")
    return Path(source).open(encoding="utf-8", newline="")


def finite_positive(value: str | None) -> float | None:
    """Return a finite positive numeric catalog value, otherwise None."""
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def calculate_rxfi(peak_value: str | None, background_value: str | None) -> float | None:
    """Return frozen RXFI for valid catalog values, otherwise None.

    RXFI is defined once here so catalog inspection and downstream analysis use
    the same positive, finite-value eligibility rule.
    """
    peak = finite_positive(peak_value)
    background = finite_positive(background_value)
    if peak is None or background is None:
        return None
    return (peak - background) / background


def quantile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated quantile without third-party packages."""
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def print_schema(columns: Iterable[str]) -> None:
    print("RXFI-related schema:")
    for column in RXFI_COLUMNS:
        print(f"  {column}: {'present' if column in columns else 'MISSING'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_CSV_URL,
        help="Local CSV path or official NOAA/NCEI CSV URL.",
    )
    args = parser.parse_args()

    with open_csv(args.source) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_columns = REQUIRED_COLUMNS.difference(columns)
        if missing_columns:
            raise SystemExit(f"Missing required column(s): {', '.join(sorted(missing_columns))}")
        print(f"Source: {args.source}")
        print_schema(columns)

        total = 0
        missing_or_invalid = 0
        rxfi_values: list[float] = []
        duplicate_keys: set[tuple[str, str]] = set()
        seen_keys: set[tuple[str, str]] = set()
        saturated = 0

        for row in reader:
            total += 1
            key = (row.get("flare_id", ""), row.get("time", ""))
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)

            if row.get("peak_saturated", "").strip() in {"1", "true", "True"}:
                saturated += 1

            rxfi = calculate_rxfi(row.get("xrsb_irrad"), row.get("background_irrad"))
            if rxfi is None:
                missing_or_invalid += 1
                continue
            rxfi_values.append(rxfi)

    rxfi_values.sort()
    print(f"\nEvents: {total}")
    print(f"Valid RXFI: {len(rxfi_values)}")
    print(f"Missing/invalid peak or background: {missing_or_invalid}")
    print(f"Duplicate (flare_id, time) keys: {len(duplicate_keys)}")
    print(f"Peak-saturated events: {saturated}")
    if not rxfi_values:
        return 0

    print("\nCandidate RXFI = (xrsb_irrad - background_irrad) / background_irrad")
    print(f"min: {rxfi_values[0]:.8g}")
    print(f"median: {statistics.median(rxfi_values):.8g}")
    print(f"mean: {statistics.fmean(rxfi_values):.8g}")
    print(f"max: {rxfi_values[-1]:.8g}")
    print("quantiles:")
    for probability in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99):
        print(f"  p{probability:.0%}: {quantile(rxfi_values, probability):.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
