"""Add event-level legacy RXFI maxima to the four fixed split target tables."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

WINDOW = timedelta(hours=24)
SPLITS = ("train", "validation", "leaky_validation", "test")
LABEL_FIELDS = (
    "max_old_rxfi",
    "max_old_rxfi_flare_id",
    "old_rxfi_event_count_24h",
    "valid_old_rxfi_event_count_24h",
    "old_rxfi_target_status",
)
OLD_RXFI_FORMULA = "catalog xrsb_irrad / old_background_xrsb_flux"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_timestamp(value: str) -> datetime | None:
    """Parse a timestamp and interpret timezone-naive NOAA/HF values as UTC."""
    if not value.strip():
        return None
    try:
        timestamp = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)


def finite_number(value: str) -> float | None:
    """Parse any finite numeric RXFI value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_catalog(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and validate the enriched event catalog and its generation manifest."""
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("catalog", {}).get("sha256") != sha256_file(Path(manifest["catalog"]["path"])):
        raise ValueError("The enriched catalog manifest does not match its NOAA source catalog")
    if manifest.get("method", {}).get("rxfi") != OLD_RXFI_FORMULA:
        raise ValueError("The enriched catalog does not declare the expected old RXFI formula")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"start_time", "time", "flare_id", "old_rxfi", "old_background_status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Enriched catalog is missing columns: {', '.join(missing)}")
        seen: dict[tuple[str, str], tuple[str, str, str]] = {}
        for row in reader:
            start = parse_timestamp(row["start_time"])
            if start is None:
                continue
            key = (row["flare_id"].strip(), row["time"].strip())
            signature = (row["start_time"], row["old_background_status"], row["old_rxfi"])
            if key in seen:
                if seen[key] != signature:
                    raise ValueError(f"Conflicting duplicate catalog event: {key}")
                continue
            seen[key] = signature
            rows.append({
                "start_time": start,
                "start_seconds": int(start.timestamp()),
                "flare_id": row["flare_id"].strip(),
                "rxfi": finite_number(row["old_rxfi"]),
                "status": row["old_background_status"].strip(),
            })
    rows.sort(key=lambda event: (event["start_seconds"], event["flare_id"]))
    return rows, manifest


def window_target(
    events: list[dict[str, Any]],
    event_starts: list[int],
    timestamp: datetime,
) -> dict[str, str | int]:
    """Aggregate valid old RXFI events whose catalog start falls in the window."""
    start = int(timestamp.timestamp())
    end = int((timestamp + WINDOW).timestamp())
    left, right = bisect.bisect_left(event_starts, start), bisect.bisect_left(event_starts, end)
    candidates = events[left:right]
    valid = [event for event in candidates if event["status"] == "ok" and event["rxfi"] is not None]
    if not candidates:
        return {
            "max_old_rxfi": "",
            "max_old_rxfi_flare_id": "",
            "old_rxfi_event_count_24h": 0,
            "valid_old_rxfi_event_count_24h": 0,
            "old_rxfi_target_status": "no_catalog_flare",
        }
    if not valid:
        return {
            "max_old_rxfi": "",
            "max_old_rxfi_flare_id": "",
            "old_rxfi_event_count_24h": len(candidates),
            "valid_old_rxfi_event_count_24h": 0,
            "old_rxfi_target_status": "no_valid_old_rxfi",
        }
    maximum = min(valid, key=lambda event: (-event["rxfi"], event["start_seconds"], event["flare_id"]))
    return {
        "max_old_rxfi": format(maximum["rxfi"], ".12g"),
        "max_old_rxfi_flare_id": maximum["flare_id"],
        "old_rxfi_event_count_24h": len(candidates),
        "valid_old_rxfi_event_count_24h": len(valid),
        "old_rxfi_target_status": "ok",
    }


def read_split(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a generated split target file without changing its source columns."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"Split target file has no timestamp column: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], overwrite: bool) -> None:
    """Write one augmented split through a temporary file to avoid partial CSVs."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Create augmented split copies, QC, and a provenance manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/derived/noaa_flare_catalog_old_background_rxfi.csv"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/old_rxfi_targets"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    events, catalog_manifest = read_catalog(args.catalog)
    event_starts = [event["start_seconds"] for event in events]
    split_hashes: dict[str, str] = {}
    qc_rows: list[dict[str, Any]] = []
    output_fields: dict[str, list[str]] = {}
    output_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        source_path = args.input_dir / f"{split}_targets.csv"
        rows, source_fields = read_split(source_path)
        if any(field in source_fields for field in LABEL_FIELDS) and not args.overwrite:
            raise ValueError(f"Input already contains old RXFI columns: {source_path}")
        split_hashes[split] = sha256_file(source_path)
        counts: Counter[str] = Counter()
        augmented: list[dict[str, Any]] = []
        for row in rows:
            timestamp = parse_timestamp(row["timestamp"])
            if timestamp is None:
                raise ValueError(f"Invalid timestamp in {source_path}: {row['timestamp']}")
            target = window_target(events, event_starts, timestamp)
            counts[str(target["old_rxfi_target_status"])] += 1
            output = {**row, **target}
            augmented.append(output)
        output_fields[split] = source_fields + [field for field in LABEL_FIELDS if field not in source_fields]
        output_rows[split] = augmented
        qc_rows.append({
            "split": split,
            "rows": len(rows),
            "windows_with_catalog_events": counts["ok"] + counts["no_valid_old_rxfi"],
            "windows_with_valid_max_old_rxfi": counts["ok"],
            "windows_without_catalog_events": counts["no_catalog_flare"],
            "windows_with_events_but_no_valid_old_rxfi": counts["no_valid_old_rxfi"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_csv(args.output_dir / f"{split}_targets.csv", output_rows[split], output_fields[split], args.overwrite)
    qc_path = args.output_dir / "target_quality_control.csv"
    write_csv(qc_path, qc_rows, list(qc_rows[0]), args.overwrite)
    metadata = {
        "target": "max_old_rxfi",
        "window": "[timestamp, timestamp + 24 hours)",
        "window_event_assignment": "catalog start_time",
        "aggregation": "maximum finite old_rxfi among events with old_background_status == ok",
        "event_old_rxfi_formula": OLD_RXFI_FORMULA,
        "no_event_or_no_valid_rxfi": "missing value; never imputed to zero",
        "tie_break": "earliest flare start_time, then lexical flare_id",
        "catalog": {"path": str(args.catalog), "sha256": sha256_file(args.catalog), "source_catalog_sha256": catalog_manifest["catalog"]["sha256"]},
        "split_inputs": [
            {"split": split, "path": str(args.input_dir / f"{split}_targets.csv"), "sha256": digest}
            for split, digest in split_hashes.items()
        ],
        "split_outputs": [str(args.output_dir / f"{split}_targets.csv") for split in SPLITS],
        "catalog_background_method": catalog_manifest["method"],
        "quality_control": str(qc_path),
        "split_rows": {row["split"]: row["rows"] for row in qc_rows},
    }
    metadata_path = args.output_dir / "target_generation_metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {metadata_path}")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote old RXFI split targets to {}", args.output_dir)
    logger.info("QC by split: {}", qc_rows)


if __name__ == "__main__":
    main()
