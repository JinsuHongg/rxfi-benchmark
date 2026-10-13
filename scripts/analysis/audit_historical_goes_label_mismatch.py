#!/usr/bin/env python3
"""Audit legacy GOES labels against immutable NOAA 24-hour derived targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DERIVED_DIR = Path("data/derived")
DEFAULT_OUTPUT_DIR = Path("outputs/historical_label_mismatch_audit")
SPLITS = ("train", "validation", "test", "leaky_validation")
ORDER = ("FQ", "A", "B", "C", "M", "X")
RANK = {name: index for index, name in enumerate(ORDER)}


def letter(value: str) -> str:
    value = value.strip().upper()
    return "FQ" if value == "FQ" else value[:1] if value[:1] in RANK else "other"


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (index - lo)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            result[order[k]] = rank
        i = j + 1
    return result


def correlation(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if len(x) < 2:
        return None, None
    def pearson(a: list[float], b: list[float]) -> float | None:
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        divisor = math.sqrt(sum((v-ma)**2 for v in a) * sum((v-mb)**2 for v in b))
        return sum((u-ma)*(v-mb) for u, v in zip(a, b)) / divisor if divisor else None
    return pearson(x, y), pearson(ranks(x), ranks(y))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_svg(path: Path, body: list[str], title: str, width: int = 900, height: int = 500) -> None:
    style = '<style>text{font-family:Arial,sans-serif;fill:#17212b}.title{font-size:18px;font-weight:bold}.label{font-size:12px}.small{font-size:10px}.axis{stroke:#425466}.cell{stroke:#fff}</style>'
    path.write_text('\n'.join(['<?xml version="1.0" encoding="UTF-8"?>', f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', style, f'<text x="35" y="28" class="title">{title}</text>', *body, '</svg>']), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    by_split: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        path = args.derived_dir / f"{split}_targets.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["split"] = split
        by_split[split] = rows
        all_rows.extend(rows)

    confusion = Counter((letter(row["max_goes_class"]), letter(row["max_flare_class"])) for row in all_rows)
    raw_rows = [{"legacy_class": legacy, "regenerated_class": generated, "count": confusion[(legacy, generated)]} for legacy in ORDER for generated in ORDER]
    normalized_rows = [{"legacy_class": legacy, "regenerated_class": generated, "row_percent": 100 * confusion[(legacy, generated)] / sum(confusion[(legacy, item)] for item in ORDER) if sum(confusion[(legacy, item)] for item in ORDER) else 0.0} for legacy in ORDER for generated in ORDER]
    write_csv(args.output_dir / "class_confusion_raw.csv", raw_rows)
    write_csv(args.output_dir / "class_confusion_row_normalized.csv", normalized_rows)

    summary_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {"adjacent": [], "severe": [], "inclusion": []}
    scatter_x: list[float] = []
    scatter_y: list[float] = []
    for split, rows in by_split.items():
        distances = [abs(RANK[letter(row["max_goes_class"])] - RANK[letter(row["max_flare_class"])]) for row in rows]
        exact = sum(distance == 0 for distance in distances)
        adjacent = sum(distance == 1 for distance in distances)
        severe = sum(distance >= 2 for distance in distances)
        summary_rows.append({"split": split, "samples": len(rows), "exact_agreement": exact, "exact_percent": 100*exact/len(rows), "adjacent_disagreement": adjacent, "adjacent_percent": 100*adjacent/len(rows), "severe_disagreement": severe, "severe_percent": 100*severe/len(rows)})
        by_year: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows: by_year[row["timestamp"][:4]].append(row)
        for year, items in sorted(by_year.items()):
            dist = [abs(RANK[letter(x["max_goes_class"])] - RANK[letter(x["max_flare_class"])]) for x in items]
            year_rows.append({"split": split, "year": year, "samples": len(items), "exact_percent": 100*sum(d==0 for d in dist)/len(items), "adjacent_percent": 100*sum(d==1 for d in dist)/len(items), "severe_percent": 100*sum(d>=2 for d in dist)/len(items)})
        for legacy in ORDER:
            items = [row for row in rows if letter(row["max_goes_class"]) == legacy]
            if items:
                dist = [abs(RANK[legacy]-RANK[letter(row["max_flare_class"])]) for row in items]
                legacy_rows.append({"split": split, "legacy_class": legacy, "samples": len(items), "exact_percent": 100*sum(d==0 for d in dist)/len(items), "adjacent_percent": 100*sum(d==1 for d in dist)/len(items), "severe_percent": 100*sum(d>=2 for d in dist)/len(items)})
        log_legacy: list[float] = []; log_new: list[float] = []; abs_diff: list[float] = []
        categories = Counter()
        for row, distance in zip(rows, distances):
            old, new = letter(row["max_goes_class"]), letter(row["max_flare_class"])
            if old != "FQ" and new == "FQ" or old == "FQ" and new != "FQ": category = "no_flare_flare_inclusion_disagreement"
            elif distance == 0: category = "same_ordinal_class"
            elif old == new: category = "same_letter_different_numeric_class"
            elif distance == 1: category = "adjacent_ordinal_class_shift"
            else: category = "severe_ordinal_class_shift"
            categories[category] += 1
            record = {"split": split, "timestamp": row["timestamp"], "legacy_class": row["max_goes_class"], "regenerated_class": row["max_flare_class"], "legacy_max_intensity": row["max_intensity"], "regenerated_max_peak_flux": row["max_peak_flux"], "regenerated_flare_id": row["max_peak_flare_id"], "flare_count_24h": row["flare_count_24h"], "ordinal_distance": distance, "legacy_event_identity": "not available in split CSV"}
            if category == "adjacent_ordinal_class_shift" and len(examples["adjacent"]) < 10: examples["adjacent"].append(record)
            if distance >= 2 and len(examples["severe"]) < 10: examples["severe"].append(record)
            if category == "no_flare_flare_inclusion_disagreement" and len(examples["inclusion"]) < 10: examples["inclusion"].append(record)
            old_flux, new_flux = float(row["max_intensity"]), float(row["max_peak_flux"])
            if old_flux > 0 and new_flux > 0:
                log_legacy.append(math.log10(old_flux)); log_new.append(math.log10(new_flux)); abs_diff.append(abs(math.log10(old_flux)-math.log10(new_flux)))
                scatter_x.append(math.log10(old_flux)); scatter_y.append(math.log10(new_flux))
        for category, count in categories.items(): category_rows.append({"split": split, "category": category, "count": count, "percent": 100*count/len(rows)})
        pearson, spearman = correlation(log_legacy, log_new)
        flux_rows.append({"split": split, "comparable_positive_flux_rows": len(abs_diff), "pearson_log10_flux": pearson, "spearman_log10_flux": spearman, "median_absolute_log10_difference": percentile(abs_diff, .5), "p90_absolute_log10_difference": percentile(abs_diff, .9), "p95_absolute_log10_difference": percentile(abs_diff, .95)})
    write_csv(args.output_dir / "agreement_summary_by_split.csv", summary_rows)
    write_csv(args.output_dir / "agreement_by_year.csv", year_rows)
    write_csv(args.output_dir / "agreement_by_legacy_class.csv", legacy_rows)
    write_csv(args.output_dir / "cause_categories.csv", category_rows)
    write_csv(args.output_dir / "continuous_flux_comparison.csv", flux_rows)
    for name, rows in examples.items(): write_csv(args.output_dir / f"examples_{name}_disagreements.csv", rows)

    # Minimal confusion heatmap, aggregated across splits.
    body=[]; max_count=max(confusion.values()) or 1; x0,y0,size=180,80,55
    for i, legacy in enumerate(ORDER):
        body.append(f'<text x="{x0-8}" y="{y0+i*size+33}" text-anchor="end" class="label">{legacy}</text>')
        for j, generated in enumerate(ORDER):
            count=confusion[(legacy,generated)]; shade=int(235-180*count/max_count)
            body.append(f'<rect x="{x0+j*size}" y="{y0+i*size}" width="{size}" height="{size}" fill="rgb({shade},{shade+10},{255})" class="cell"/><text x="{x0+j*size+size/2}" y="{y0+i*size+30}" text-anchor="middle" class="small">{count:,}</text>')
    for j, name in enumerate(ORDER): body.append(f'<text x="{x0+j*size+size/2}" y="{y0-10}" text-anchor="middle" class="label">{name}</text>')
    body += ['<text x="345" y="440" text-anchor="middle" class="label">Legacy class (rows) versus regenerated NOAA class (columns)</text>']
    save_svg(args.output_dir / "figure_class_confusion.svg", body, "Legacy versus regenerated NOAA class confusion")
    # Agreement by year plot (train/validation/leaky combined historical, test separately).
    aggregate=defaultdict(list)
    for row in all_rows: aggregate[row["timestamp"][:4]].append(row)
    body=[]; years=sorted(aggregate); x0,y0,w,h=80,65,760,330
    for i,year in enumerate(years):
        items=aggregate[year]; rate=100*sum(letter(x["max_goes_class"])==letter(x["max_flare_class"]) for x in items)/len(items); bw=w/len(years)*.7; x=x0+i*w/len(years)+(w/len(years)-bw)/2; bh=h*rate/100
        body.append(f'<rect x="{x:.1f}" y="{y0+h-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#2b6cb0"/><text x="{x+bw/2:.1f}" y="{y0+h+15}" text-anchor="middle" class="small">{year}</text>')
    body += [f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>', f'<text x="{x0}" y="{y0+h+40}" class="small">Exact ordinal-letter agreement (%)</text>']
    save_svg(args.output_dir / "figure_agreement_by_year.svg", body, "Exact legacy/regenerated class agreement by year")
    # Deterministic scatter (every Nth observation), log-log data already transformed.
    body=[]; n=max(1,len(scatter_x)//18000); xmin,xmax=min(scatter_x),max(scatter_x); ymin,ymax=min(scatter_y),max(scatter_y); x0,y0,w,h=95,60,700,330
    for i in range(0,len(scatter_x),n):
        x=x0+w*(scatter_x[i]-xmin)/(xmax-xmin); y=y0+h*(1-(scatter_y[i]-ymin)/(ymax-ymin)); body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" fill="#2b6cb0" fill-opacity=".2"/>')
    body += [f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0}" stroke="#d53f8c"/>','<text x="445" y="425" text-anchor="middle" class="label">log10 legacy max_intensity</text><text x="20" y="225" transform="rotate(-90 20 225)" class="label">log10 regenerated max_peak_flux</text>']
    save_svg(args.output_dir / "figure_legacy_vs_regenerated_flux.svg", body, "Legacy versus regenerated maximum peak flux")
    (args.output_dir / "audit_metadata.json").write_text(json.dumps({"legacy_event_identity": "not available in split CSVs", "window_contract": "[t, t + 24 hours)", "rank_order": ORDER, "timing_code_found": False}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
