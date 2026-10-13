#!/usr/bin/env python3
"""Fill FQ max_peak_flux from official NOAA XRS-B 1-minute averages.

This is a deterministic refinement of the already-confirmed [t, t + 24h)
contract.  It never changes flare-event or RXFI-derived columns.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import urllib.request
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date

SPLITS = ("train", "validation", "test", "leaky_validation")
SOURCE_HASHES = {
    "train": "2ec7b8f39367f8340a39889bc66525aff303410d7b7ce6c12a55ea346b55e865",
    "validation": "803d2e5584fe9bbe23bc02cbed1b06fb47520e4863c2b22b5f09f9d5c654c658",
    "test": "40ddef01aebe23e5ee460717a08b7392827eacca2852af074d5f1533f59ebd4b",
    "leaky_validation": "03134a82a53891d25761774c5aad52f77e01673195f7cfd28c0dc061bfe5849e",
}
# NOAA XRS science-quality ReadMe Table 11. Entries are effective at UTC.
PRIMARY_SCHEDULE = (
    ("2010-01-01T00:00:00", 14, None), ("2010-09-01T00:00:00", 15, 14),
    ("2012-10-23T16:00:00", 14, 15), ("2012-11-19T16:31:00", 15, 14),
    ("2015-01-26T16:01:00", 13, 15), ("2015-05-21T18:00:00", 14, 13),
    ("2015-06-09T16:25:00", 15, 14), ("2016-05-03T13:00:00", 13, 14),
    ("2016-05-12T17:30:00", 14, 13), ("2016-05-16T17:00:00", 15, 13),
    ("2017-02-07T00:00:00", 16, 15), ("2018-06-01T00:00:00", 17, 16),
    ("2023-01-10T00:00:00", 18, 17),
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

def source_for(timestamp: datetime) -> tuple[int, int | None]:
    selected = PRIMARY_SCHEDULE[0]
    for entry in PRIMARY_SCHEDULE:
        if timestamp >= parse_time(entry[0]): selected = entry
        else: break
    return selected[1], selected[2]

def url_for(satellite: int, year: int) -> str:
    filename = f"sci_xrsf-l2-avg1m_g{satellite}_y{year}_v2-2-1.nc"
    if satellite <= 15:
        return f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes{satellite}/xrsf-l2-avg1m_science/{filename}"
    return f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{satellite}/l2/data/xrsf-l2-avg1m_science/{filename}"

class XrsCache:
    def __init__(self, root: Path, allow_download: bool):
        self.root, self.allow_download, self.series = root, allow_download, {}

    def load(self, satellite: int, year: int) -> tuple[np.ndarray, np.ndarray] | None:
        key = (satellite, year)
        if key in self.series: return self.series[key]
        raw = self.root / f"g{satellite}" / f"{year}.nc"
        if not raw.exists():
            if not self.allow_download: return None
            raw.parent.mkdir(parents=True, exist_ok=True)
            try: urllib.request.urlretrieve(url_for(satellite, year), raw)
            except Exception:
                raw.unlink(missing_ok=True); return None
        try:
            with Dataset(raw) as dataset:
                required = {"time", "xrsb_flux", "xrsb_flag"}
                if not required.issubset(dataset.variables): raise ValueError(f"Unexpected fields in {raw}")
                time_values = num2date(dataset["time"][:], dataset["time"].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                seconds = np.asarray([int(value.replace(tzinfo=timezone.utc).timestamp()) for value in time_values], dtype=np.int64)
                flux = np.asarray(dataset["xrsb_flux"][:], dtype=float)
                flags = np.asarray(dataset["xrsb_flag"][:])
                valid = np.isfinite(flux) & (flux > 0) & (flags == 0)
                result = (seconds[valid], flux[valid])
        except Exception:
            return None
        self.series[key] = result
        return result

    def window(self, satellite: int, start: datetime, end: datetime) -> np.ndarray | None:
        pieces: list[np.ndarray] = []
        for year in range(start.year, end.year + 1):
            item = self.load(satellite, year)
            if item is None: return None
            seconds, flux = item; lower, upper = int(start.timestamp()), int(end.timestamp())
            left, right = np.searchsorted(seconds, lower), np.searchsorted(seconds, upper)
            pieces.append(flux[left:right])
        return np.concatenate(pieces) if pieces else np.empty(0)

def percentile(values: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(values), p))

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-dir", type=Path, default=Path("/mnt/storage/surya/index_data"))
    parser.add_argument("--derived-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/noaa_xrs_avg1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/no_flare_max_peak_flux_refinement"))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = XrsCache(args.cache_dir, args.allow_download)
    qc_rows: list[dict[str, object]] = []; summary: dict[str, object] = {"source_sha256": {}, "splits": {}}
    all_no_flare, all_log, all_valid_log = [], [], []
    for split in SPLITS:
        source, derived = args.source_split_dir / f"{split}.csv", args.derived_dir / f"{split}_targets.csv"
        if sha256(source) != SOURCE_HASHES[split]: raise RuntimeError(f"Frozen source hash changed: {source}")
        with derived.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle)); fields = list(rows[0])
        original = [dict(row) for row in rows]
        if "max_peak_flux_source" not in fields: fields += ["max_peak_flux_source", "xrsb_1min_valid_count_24h", "xrsb_1min_coverage_fraction_24h"]
        changed = missing = 0
        for row in rows:
            if row["max_flare_class"] != "FQ":
                row.update({"max_peak_flux_source":"flare_summary", "xrsb_1min_valid_count_24h":"", "xrsb_1min_coverage_fraction_24h":""})
            else:
                start, end = parse_time(row["timestamp"]), parse_time(row["timestamp"]) + timedelta(hours=24)
                primary, secondary = source_for(start); values = cache.window(primary, start, end); used = f"goes{primary}_primary"
                if values is None or len(values) == 0:
                    values = cache.window(secondary, start, end) if secondary else None; used = f"goes{secondary}_secondary" if secondary else "missing"
                valid_count = 0 if values is None else len(values)
                coverage = valid_count / 1440
                if valid_count:
                    row["max_peak_flux"] = format(float(np.max(values)), ".12g"); row["max_peak_flux_source"] = "xrsb_1min_no_flare"
                    all_no_flare.append(float(row["max_peak_flux"])); all_log.append(math.log10(float(row["max_peak_flux"]))); changed += 1
                else:
                    row["max_peak_flux"] = ""; row["max_peak_flux_source"] = "missing"; missing += 1
                row["xrsb_1min_valid_count_24h"] = str(valid_count); row["xrsb_1min_coverage_fraction_24h"] = format(coverage, ".8f")
                qc_rows.append({"split":split,"timestamp":row["timestamp"],"primary_satellite":primary,"secondary_satellite":secondary or "","source_used":used,"valid_count_24h":valid_count,"coverage_fraction_24h":coverage,"max_peak_flux":row["max_peak_flux"],"status":row["max_peak_flux_source"]})
            try:
                if float(row["max_peak_flux"]) > 0: all_valid_log.append(math.log10(float(row["max_peak_flux"])))
            except ValueError: pass
        for before, after in zip(original, rows):
            if before["max_flare_class"] != "FQ" and before["max_peak_flux"] != after["max_peak_flux"]: raise RuntimeError("Flare max_peak_flux changed")
            for column in ("max_flare_class", "cumulative_peak_flux", "max_rxfi", "cumulative_rxfi"):
                if before[column] != after[column]: raise RuntimeError(f"Immutable target changed: {column}")
        backup = args.output_dir / "previous_targets" / derived.name; backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists(): shutil.copy2(derived, backup)
        with derived.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        summary["source_sha256"][split] = sha256(source); summary["splits"][split] = {"rows":len(rows),"fq_rows":sum(r["max_flare_class"]=="FQ" for r in rows),"assigned":changed,"missing":missing}
    write_csv(args.output_dir / "no_flare_xrsb_window_qc.csv", qc_rows)
    coverage_rows = []
    for split in SPLITS:
        values = [float(row["coverage_fraction_24h"]) for row in qc_rows if row["split"] == split]
        statuses = Counter(row["status"] for row in qc_rows if row["split"] == split)
        coverage_rows.append({"split": split, "fq_windows": len(values), "assigned": statuses["xrsb_1min_no_flare"], "missing": statuses["missing"], "coverage_min": min(values), "coverage_median": percentile(values, .5), "coverage_mean": float(np.mean(values)), "coverage_p05": percentile(values, .05), "coverage_max": max(values)})
    write_csv(args.output_dir / "no_flare_xrsb_coverage_summary.csv", coverage_rows)
    comparison_rows = []
    for split in SPLITS:
        before_path = args.output_dir / "previous_targets" / f"{split}_targets.csv"
        with before_path.open(encoding="utf-8", newline="") as handle: before = list(csv.DictReader(handle))
        with (args.derived_dir / f"{split}_targets.csv").open(encoding="utf-8", newline="") as handle: after = list(csv.DictReader(handle))
        fq_before = [float(row["max_peak_flux"]) for row in before if row["max_flare_class"] == "FQ"]
        fq_after = [float(row["max_peak_flux"]) for row in after if row["max_flare_class"] == "FQ" and row["max_peak_flux"]]
        comparison_rows.append({"split":split,"flare_windows":sum(row["max_flare_class"] != "FQ" for row in after),"flare_max_peak_flux_changed":sum(a["max_flare_class"] != "FQ" and a["max_peak_flux"] != b["max_peak_flux"] for a,b in zip(before,after)),"fq_windows":len(fq_before),"fq_zero_before":sum(value == 0 for value in fq_before),"fq_positive_after":sum(value > 0 for value in fq_after),"fq_missing_after":len(fq_before)-len(fq_after)})
    write_csv(args.output_dir / "before_after_max_peak_flux_summary.csv", comparison_rows)
    stats = {"count":len(all_no_flare),"min":min(all_no_flare),"q1":percentile(all_no_flare,.25),"median":percentile(all_no_flare,.5),"mean":float(np.mean(all_no_flare)),"q3":percentile(all_no_flare,.75),"p90":percentile(all_no_flare,.9),"p95":percentile(all_no_flare,.95),"p99":percentile(all_no_flare,.99),"max":max(all_no_flare)}
    summary.update({"product":"NOAA/NCEI science-quality XRS L2 avg1m, v2-2-1","field":"xrsb_flux","units":"W/m2","valid_rule":"finite, positive, xrsb_flag == 0","window":"[t, t + 24 hours)","no_flare_statistics":stats,"full_log10_valid_count":len(all_valid_log)})
    (args.output_dir / "refinement_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10,4)); axes[0].hist(all_no_flare, bins=60); axes[0].set(xlabel="FQ 24h max XRS-B (W/m²)", ylabel="samples"); axes[1].hist(all_valid_log, bins=60); axes[1].set(xlabel="log10(max_peak_flux), all valid samples", ylabel="samples"); fig.tight_layout(); fig.savefig(args.output_dir / "no_flare_max_flux_distributions.svg", format="svg"); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
