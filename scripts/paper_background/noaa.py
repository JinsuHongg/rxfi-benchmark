"""Pinned NOAA/NCEI source URLs, downloads, and XRS L2 NetCDF reader."""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/"
GOES_R_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
SUPPORTED_SATELLITES = {13, 14, 15, 16, 17, 18, 19}


class InputDownloadError(RuntimeError):
    """Raised when a NOAA input cannot be retrieved and verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog_url(config: dict[str, Any]) -> str:
    """Return the catalog release URL pinned in the run configuration."""
    catalog = config["catalog"]
    filename = catalog["filename"]
    url = catalog["url"]
    if not url.endswith(filename):
        raise ValueError("catalog URL must end with the pinned catalog filename")
    return url


def l2_url(satellite: int, year: int, version: str) -> str:
    """Build the official annual science-quality avg1m URL."""
    satellite, year = int(satellite), int(year)
    if satellite not in SUPPORTED_SATELLITES:
        raise ValueError(f"unsupported GOES satellite: {satellite}")
    if year < 1995 or year > 2100:
        raise ValueError(f"invalid XRS year: {year}")
    filename = f"sci_xrsf-l2-avg1m_g{satellite}_y{year}_{version}.nc"
    if satellite <= 15:
        return f"{LEGACY_BASE}goes{satellite}/xrsf-l2-avg1m_science/{filename}"
    return f"{GOES_R_BASE}goes{satellite}/l2/data/xrsf-l2-avg1m_science/{filename}"


def cache_path(cache_root: Path, satellite: int, year: int, version: str) -> Path:
    """Return a version-specific cache path for one satellite/year input."""
    return Path(cache_root) / "xrs_l2" / version / f"g{int(satellite)}" / f"{int(year)}.nc"


def download_atomic(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Stream a URL to a temporary file and publish only verified bytes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rxfi-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as output:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type", "").lower()
            if status != 200:
                raise InputDownloadError(f"HTTP {status} retrieving {url}")
            if "text/html" in content_type:
                raise InputDownloadError(f"NOAA returned HTML instead of data for {url}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                byte_count += len(block)
            output.flush()
            os.fsync(output.fileno())
        checksum = digest.hexdigest()
        if byte_count == 0:
            raise InputDownloadError(f"empty response retrieving {url}")
        if expected_size is not None and byte_count != int(expected_size):
            raise InputDownloadError(
                f"size mismatch for {url}: expected {expected_size}, received {byte_count}"
            )
        if expected_sha256 is not None and checksum.lower() != expected_sha256.lower():
            raise InputDownloadError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, received {checksum}"
            )
        temporary.replace(destination)
        return {
            "url": url,
            "path": str(destination),
            "size_bytes": byte_count,
            "sha256": checksum,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, InputDownloadError):
            raise
        if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError, EOFError)):
            raise InputDownloadError(f"failed to retrieve complete input from {url}: {exc}") from exc
        raise InputDownloadError(f"failed to retrieve {url}: {type(exc).__name__}: {exc}") from exc


def read_l2_year(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read and validate one science-quality XRS L2 `avg1m` NetCDF file."""
    path = Path(path)
    with xr.open_dataset(path, engine="h5netcdf", decode_times=True) as dataset:
        required = {"time", "xrsb_flux", "xrsb_flag"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"missing required NetCDF variable(s) in {path}: {', '.join(missing)}")
        raw_time = np.asarray(dataset["time"].values)
        if not np.issubdtype(raw_time.dtype, np.datetime64):
            raise ValueError(f"time coordinate is not decoded to UTC datetime64 in {path}: {raw_time.dtype}")
        seconds = raw_time.astype("datetime64[s]").astype(np.int64).reshape(-1)
        flux = np.asarray(dataset["xrsb_flux"].values, dtype=float).reshape(-1)
        flags = np.asarray(dataset["xrsb_flag"].values).reshape(-1)
        if not (len(seconds) == len(flux) == len(flags)):
            raise ValueError(f"time, xrsb_flux, and xrsb_flag lengths differ in {path}")
        attrs = dict(dataset.attrs)
    order = np.argsort(seconds, kind="stable")
    seconds, flux, flags = seconds[order], flux[order], flags[order]
    if len(seconds) > 1 and np.any(seconds[1:] < seconds[:-1]):
        raise ValueError(f"timestamps could not be sorted in {path}")
    return seconds, flux, flags, {
        "filename": path.name,
        "sample_count": int(len(seconds)),
        "time_start_utc": str(raw_time.min()) if len(raw_time) else None,
        "time_end_utc": str(raw_time.max()) if len(raw_time) else None,
        "global_attributes": {key: str(value) for key, value in attrs.items()},
    }


def verify_input_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict[str, Any]:
    """Verify a local cache entry against the immutable NOAA input lock."""
    if not path.is_file():
        raise FileNotFoundError(f"required NOAA input is missing: {path}")
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if expected_size is not None and actual_size != int(expected_size):
        raise InputDownloadError(f"size mismatch for {path}: expected {expected_size}, received {actual_size}")
    if actual_sha.lower() != expected_sha256.lower():
        raise InputDownloadError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, received {actual_sha}")
    return {"path": str(path), "size_bytes": actual_size, "sha256": actual_sha}


def find_cached_input(
    cache_root: Path,
    satellite: int,
    year: int,
    version: str,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Find a checksum-matching versioned or legacy-layout local cache file.

    The legacy path is accepted only if it matches the exact hash and size in
    the input lock, preventing accidental reuse across product versions.
    """
    versioned = cache_path(cache_root, satellite, year, version)
    legacy = Path(cache_root) / "xrs_l2" / f"g{int(satellite)}" / f"{int(year)}.nc"
    errors: list[str] = []
    for candidate, is_legacy in ((versioned, False), (legacy, True)):
        if not candidate.is_file():
            continue
        try:
            record = verify_input_file(candidate, expected_sha256, expected_size)
            record["legacy_layout"] = is_legacy
            return candidate, record
        except InputDownloadError as exc:
            errors.append(str(exc))
    if errors:
        raise InputDownloadError("no cached copy matches the pinned NOAA input: " + "; ".join(errors))
    raise FileNotFoundError(
        f"required NOAA input is not cached: satellite=GOES-{satellite}, year={year}, version={version}"
    )
