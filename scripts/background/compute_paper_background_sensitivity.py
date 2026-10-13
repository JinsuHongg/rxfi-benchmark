#!/usr/bin/env python3
"""Run the checksum-pinned paper-based 24-hour XRS background calculation."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper_background.runner import run_calculation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_background_sensitivity.json"))
    parser.add_argument("--catalog", type=Path, help="local copy of the exact catalog release pinned in config")
    parser.add_argument("--cache-dir", type=Path, help="NOAA input cache root (default: data/raw/noaa_paper_background)")
    parser.add_argument("--output-root", type=Path, help="run directory parent (default: config output.directory)")
    parser.add_argument("--offline", action="store_true", help="require all locked inputs to exist locally")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing directory for this exact fingerprint")
    args = parser.parse_args()
    result = run_calculation(
        args.config, catalog_path=args.catalog, cache_root=args.cache_dir,
        output_root=args.output_root, offline=args.offline, overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
