#!/usr/bin/env python3
"""Materialize versioned Project 3 exports from completed QR predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

from project3_export_contract import export_target, source_experiment_dir

TARGETS = ("max_peak_flux", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, action="append", help="Target to export; repeatable")
    parser.add_argument("--all", action="store_true", help="Export all completed QR targets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "project3_exports")
    args = parser.parse_args()
    if args.all == bool(args.target):
        parser.error("choose exactly one of --all or --target")
    for target in TARGETS if args.all else args.target:
        print(export_target(source_experiment_dir(ROOT, target), args.output_root, target))


if __name__ == "__main__":
    main()
