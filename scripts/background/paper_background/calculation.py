"""Pure event-level calculations for paper-based 24-hour XRS backgrounds."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

SECONDS_PER_DAY = 86_400


def deduplicate_catalog_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    """Keep the first row for each exact ``(flare_id, time)`` key."""
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    duplicates = 0
    for row in rows:
        key = (row.get("flare_id", ""), row.get("time", ""))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def valid_measurement_mask(flux: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Return finite positive XRS-B values whose quality flag is exactly zero."""
    values = np.asarray(flux, dtype=float).reshape(-1)
    quality = np.asarray(flags).reshape(-1)
    if len(values) != len(quality):
        raise ValueError("flux and quality flag arrays must have the same length")
    return np.isfinite(values) & (values > 0) & (quality == 0)


def mask_other_flare_samples(
    sample_seconds: np.ndarray,
    intervals: Sequence[tuple[int, int, str]],
    target_flare_id: str,
) -> np.ndarray:
    """Mark samples in any other flare's inclusive interval.

    Intervals are merged after removing the target event. Binary search then
    evaluates all sample timestamps without scanning every interval per sample.
    """
    seconds = np.asarray(sample_seconds, dtype=np.int64).reshape(-1)
    merged = merge_flare_intervals(intervals, excluded_flare_id=target_flare_id)
    return mask_merged_flare_intervals(seconds, merged)


def merge_flare_intervals(
    intervals: Sequence[tuple[int, int, str]],
    *,
    excluded_flare_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping inclusive intervals once for repeated event queries."""
    other = sorted(
        (int(start), int(end))
        for start, end, flare_id in intervals
        if (excluded_flare_id is None or flare_id != excluded_flare_id)
        and int(end) >= int(start)
    )
    if not other:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    merged: list[list[int]] = []
    for start, end in other:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return (
        np.asarray([interval[0] for interval in merged], dtype=np.int64),
        np.asarray([interval[1] for interval in merged], dtype=np.int64),
    )


def mask_merged_flare_intervals(
    sample_seconds: np.ndarray,
    merged_intervals: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Vectorized membership query against premerged inclusive intervals."""
    seconds = np.asarray(sample_seconds, dtype=np.int64).reshape(-1)
    starts, ends = (np.asarray(part, dtype=np.int64).reshape(-1) for part in merged_intervals)
    if len(starts) != len(ends):
        raise ValueError("merged interval start/end arrays must have equal length")
    if not len(starts) or not len(seconds):
        return np.zeros(len(seconds), dtype=bool)
    preceding = np.searchsorted(starts, seconds, side="right") - 1
    masked = preceding >= 0
    masked[masked] &= seconds[masked] <= ends[preceding[masked]]
    return masked


def index_flare_intervals(
    intervals: Sequence[tuple[int, int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition the union into disjoint segments annotated with active IDs."""
    per_id: dict[str, list[tuple[int, int]]] = {}
    for start, end, flare_id in intervals:
        start, end = int(start), int(end)
        if end >= start:
            per_id.setdefault(str(flare_id), []).append((start, end))
    changes: dict[int, list[tuple[str, int]]] = {}
    for flare_id, pieces in per_id.items():
        for start, end in _merge_plain_intervals(pieces):
            changes.setdefault(start, []).append((flare_id, 1))
            changes.setdefault(end + 1, []).append((flare_id, -1))
    if not changes:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), np.empty(0, dtype=object)
    points = sorted(changes)
    active: set[str] = set()
    starts: list[int] = []
    ends: list[int] = []
    counts: list[int] = []
    owners: list[str] = []
    for index, point in enumerate(points[:-1]):
        # Apply all changes at the left endpoint before labeling this segment.
        for flare_id, delta in changes[point]:
            if delta < 0:
                active.discard(flare_id)
        for flare_id, delta in changes[point]:
            if delta > 0:
                active.add(flare_id)
        next_point = points[index + 1]
        if next_point <= point or not active:
            continue
        starts.append(point)
        ends.append(next_point - 1)
        counts.append(len(active))
        owners.append(next(iter(active)) if len(active) == 1 else "")
    return (np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64),
            np.asarray(counts, dtype=np.int32), np.asarray(owners, dtype=object))


def _merge_plain_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def mask_indexed_flare_intervals(
    sample_seconds: np.ndarray,
    indexed_intervals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    target_flare_id: str,
) -> np.ndarray:
    """Mask samples covered by any active flare other than the target ID."""
    seconds = np.asarray(sample_seconds, dtype=np.int64).reshape(-1)
    starts, ends, counts, owners = indexed_intervals
    if not len(starts) or not len(seconds):
        return np.zeros(len(seconds), dtype=bool)
    positions = np.searchsorted(starts, seconds, side="right") - 1
    present = positions >= 0
    covered = np.zeros(len(seconds), dtype=bool)
    covered[present] = seconds[present] <= ends[positions[present]]
    result = covered.copy()
    exclusive = present & covered & (counts[np.maximum(positions, 0)] == 1)
    result[exclusive] &= owners[positions[exclusive]] != str(target_flare_id)
    return result


def calculate_event_backgrounds(
    sample_seconds: np.ndarray,
    flux: np.ndarray,
    flags: np.ndarray,
    start_seconds: int,
    start_threshold: float | None,
    peak_flux: float,
    flare_intervals: Sequence[tuple[int, int, str]],
    target_flare_id: str,
    *,
    window_seconds: int = SECONDS_PER_DAY,
    expected_minutes: int = 1_440,
    start_sample_status: str | None = None,
    merged_flare_intervals: tuple[np.ndarray, np.ndarray] | None = None,
    indexed_flare_intervals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Calculate paired 24-hour arithmetic backgrounds and fractional RXFI.

    Input arrays may contain timestamps outside the target window; selection is
    always the strict pre-start interval ``[start - window, start)``.
    """
    seconds = np.asarray(sample_seconds, dtype=np.int64).reshape(-1)
    values = np.asarray(flux, dtype=float).reshape(-1)
    quality = np.asarray(flags).reshape(-1)
    if not (len(seconds) == len(values) == len(quality)):
        raise ValueError("timestamp, flux, and quality flag arrays must have the same length")
    if window_seconds <= 0 or expected_minutes <= 0:
        raise ValueError("window_seconds and expected_minutes must be positive")
    start_seconds = int(start_seconds)
    lower = start_seconds - int(window_seconds)
    in_window = (seconds >= lower) & (seconds < start_seconds)
    seconds, values, quality = seconds[in_window], values[in_window], quality[in_window]

    valid = valid_measurement_mask(values, quality)
    valid_quality_count = int(valid.sum())
    if indexed_flare_intervals is not None:
        flare_mask = mask_indexed_flare_intervals(seconds, indexed_flare_intervals, target_flare_id)
    elif merged_flare_intervals is None:
        flare_mask = mask_other_flare_samples(seconds, flare_intervals, target_flare_id)
    else:
        # The runner may use the full premerged catalog union. The target's own
        # interval starts at t0 and cannot overlap this strict pre-start window.
        flare_mask = mask_merged_flare_intervals(seconds, merged_flare_intervals)
    other_flare_masked_count = int((valid & flare_mask).sum())
    common = valid & ~flare_mask
    uncapped_values = values[common]
    background_uncapped = float(np.mean(uncapped_values)) if len(uncapped_values) else None

    if start_sample_status is None:
        start_sample_status = "valid" if start_threshold is not None else "missing_exact_sample"
    if start_threshold is None or not math.isfinite(float(start_threshold)) or float(start_threshold) <= 0:
        background_capped = None
        capped_values = np.empty(0, dtype=float)
        capped_status = start_sample_status if start_sample_status != "valid" else "invalid_start_threshold"
        capped_removed_count: int | None = None
    else:
        threshold = float(start_threshold)
        capped_values = values[common & (values <= threshold)]
        background_capped = float(np.mean(capped_values)) if len(capped_values) else None
        capped_status = "ok" if background_capped is not None else "no_prestart_sample_at_or_below_threshold"
        capped_removed_count = int((common & (values > threshold)).sum())

    if background_uncapped is None:
        uncapped_status = "no_valid_background_samples"
    else:
        uncapped_status = "ok"
    rxfi_uncapped = (
        (float(peak_flux) - background_uncapped) / background_uncapped
        if background_uncapped is not None and background_uncapped > 0
        else None
    )
    rxfi_capped = (
        (float(peak_flux) - background_capped) / background_capped
        if background_capped is not None and background_capped > 0
        else None
    )

    return {
        "candidate_count": int(len(values)),
        "valid_quality_count": valid_quality_count,
        "other_flare_masked_count": other_flare_masked_count,
        "uncapped_retained_count": int(len(uncapped_values)),
        "capped_removed_by_threshold_count": capped_removed_count,
        "capped_retained_count": int(len(capped_values)) if start_threshold is not None else None,
        "uncapped_coverage_fraction": len(uncapped_values) / expected_minutes,
        "capped_coverage_fraction": len(capped_values) / expected_minutes if start_threshold is not None else None,
        "start_sample_status": start_sample_status,
        "uncapped_status": uncapped_status,
        "capped_status": capped_status,
        "background_uncapped": background_uncapped,
        "background_capped": background_capped,
        "rxfi_uncapped": rxfi_uncapped,
        "rxfi_capped": rxfi_capped,
    }
