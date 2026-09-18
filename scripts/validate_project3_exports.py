#!/usr/bin/env python3
"""Validate Project 3 prediction exports without changing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project3_export_contract import validate_export_directory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", help="Export target directory name; repeatable")
    parser.add_argument("--all", action="store_true", help="Validate every target under --export-root")
    parser.add_argument("--export-root", type=Path, default=ROOT / "outputs" / "project3_exports")
    args = parser.parse_args()
    if args.all == bool(args.target):
        parser.error("choose exactly one of --all or --target")
    names = sorted(path.name for path in args.export_root.iterdir() if path.is_dir()) if args.all else args.target
    print(json.dumps([validate_export_directory(args.export_root / name) for name in names], indent=2))


if __name__ == "__main__":
    main()
