"""Copyable stdlib-only example consumer for ordinal-cqr."""
from __future__ import annotations
import csv, json
from pathlib import Path

SCHEMA_VERSION = "project3_prediction_export_v1"

def load_project3_export(path: str | Path):
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {metadata.get('schema_version')!r}")
    def rows(split: str):
        with (root / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return {"validation": rows("validation"), "calibration": rows("leaky_validation"), "test": rows("test")}, metadata
