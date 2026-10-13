import sys
import http.client
import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_background.noaa import (  # noqa: E402
    InputDownloadError,
    cache_path,
    download_atomic,
    find_cached_input,
    l2_url,
    read_l2_year,
)
from paper_background.runner import run_calculation  # noqa: E402
from paper_background.calculation import (  # noqa: E402
    calculate_event_backgrounds,
    deduplicate_catalog_rows,
    mask_merged_flare_intervals,
    mask_indexed_flare_intervals,
    index_flare_intervals,
    mask_other_flare_samples,
    merge_flare_intervals,
    valid_measurement_mask,
)


def test_deduplicate_catalog_rows_removes_only_exact_keys():
    rows = [
        {"flare_id": "a", "time": "t1", "other": "first"},
        {"flare_id": "a", "time": "t1", "other": "duplicate"},
        {"flare_id": "a", "time": "t2", "other": "different time"},
        {"flare_id": "b", "time": "t1", "other": "different id"},
        {"flare_id": "", "time": "t3", "other": "blank id but unique time"},
    ]

    unique, duplicate_count = deduplicate_catalog_rows(rows)

    assert duplicate_count == 1
    assert [row["other"] for row in unique] == ["first", "different time", "different id", "blank id but unique time"]


def test_valid_measurement_mask_requires_positive_finite_flux_and_zero_flag():
    flux = np.array([1.0, 0.0, -1.0, np.nan, np.inf, 2.0])
    flags = np.array([0, 0, 0, 0, 0, 1])

    assert valid_measurement_mask(flux, flags).tolist() == [True, False, False, False, False, False]


def test_other_flare_mask_includes_both_interval_endpoints_and_ignores_target():
    seconds = np.array([99, 100, 101, 102, 103, 104, 105])
    intervals = [(100, 102, "other"), (103, 104, "target")]

    masked = mask_other_flare_samples(seconds, intervals, target_flare_id="target")

    assert masked.tolist() == [False, True, True, True, False, False, False]


def test_premerged_flare_union_masks_inclusive_intervals_without_resorting():
    seconds = np.array([99, 100, 101, 102, 103, 104, 105])
    merged = merge_flare_intervals([(100, 102, "one"), (102, 104, "two")])

    assert mask_merged_flare_intervals(seconds, merged).tolist() == [
        False, True, True, True, True, True, False,
    ]


def test_indexed_intervals_exclude_all_target_id_intervals_but_keep_overlaps():
    seconds = np.array([9, 10, 11, 12, 13, 14, 15, 16])
    indexed = index_flare_intervals([
        (10, 12, "target"), (14, 15, "target"), (11, 14, "other"),
    ])

    assert mask_indexed_flare_intervals(seconds, indexed, "target").tolist() == [
        False, False, True, True, True, True, False, False,
    ]


def test_event_background_uses_strict_prestart_window_and_inclusive_cap():
    t0 = 2 * 86_400
    seconds = np.array([t0 - 86_400, t0 - 60, t0, t0 + 60])
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    flags = np.zeros(4, dtype=int)

    result = calculate_event_backgrounds(
        seconds,
        flux,
        flags,
        start_seconds=t0,
        start_threshold=2.0,
        peak_flux=5.0,
        flare_intervals=[],
        target_flare_id="target",
    )

    assert result["uncapped_retained_count"] == 2
    assert result["capped_retained_count"] == 2
    assert result["background_uncapped"] == 1.5
    assert result["background_capped"] == 1.5
    assert result["rxfi_uncapped"] == result["rxfi_capped"] == (5.0 - 1.5) / 1.5


def test_event_background_masks_other_flare_endpoints_and_bad_quality_samples():
    t0 = 2 * 86_400
    seconds = np.array([t0 - 240, t0 - 180, t0 - 120, t0 - 60])
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    flags = np.array([0, 0, 1, 0])

    result = calculate_event_backgrounds(
        seconds,
        flux,
        flags,
        start_seconds=t0,
        start_threshold=4.0,
        peak_flux=8.0,
        flare_intervals=[(t0 - 240, t0 - 180, "other")],
        target_flare_id="target",
    )

    assert result["other_flare_masked_count"] == 2
    assert result["valid_quality_count"] == 3
    assert result["uncapped_retained_count"] == 1
    assert result["background_uncapped"] == 4.0


def test_missing_or_invalid_start_sample_only_removes_capped_estimate():
    t0 = 2 * 86_400
    seconds = np.array([t0 - 60])
    flux = np.array([1.0])
    flags = np.array([0])

    for sample_status in ("missing_exact_sample", "invalid_quality_or_flux"):
        result = calculate_event_backgrounds(
            seconds,
            flux,
            flags,
            start_seconds=t0,
            start_threshold=None,
            peak_flux=2.0,
            flare_intervals=[],
            target_flare_id="target",
            start_sample_status=sample_status,
        )
        assert result["background_uncapped"] == 1.0
        assert result["background_capped"] is None
        assert result["capped_status"] == sample_status


def test_no_prestart_samples_below_cap_has_explicit_missing_status():
    t0 = 2 * 86_400
    result = calculate_event_backgrounds(
        np.array([t0 - 60]),
        np.array([2.0]),
        np.array([0]),
        start_seconds=t0,
        start_threshold=1.0,
        peak_flux=3.0,
        flare_intervals=[],
        target_flare_id="target",
    )

    assert result["background_uncapped"] == 2.0
    assert result["background_capped"] is None
    assert result["capped_status"] == "no_prestart_sample_at_or_below_threshold"


def test_valid_negative_uncapped_rxfi_is_preserved():
    t0 = 2 * 86_400
    result = calculate_event_backgrounds(
        np.array([t0 - 60]),
        np.array([4.0]),
        np.array([0]),
        start_seconds=t0,
        start_threshold=4.0,
        peak_flux=2.0,
        flare_intervals=[],
        target_flare_id="target",
    )

    assert result["rxfi_uncapped"] == -0.5
    assert result["rxfi_capped"] == -0.5


def test_configuration_and_lock_pin_the_frozen_noaa_inputs():
    import json

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/paper_background_sensitivity.json").read_text())
    lock_path = root / config["input_lock"]
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)

    assert config["catalog"]["filename"] == "sci_xrsf-l2-flrpt_geo_s19950103_e20261012_v1-0-1.csv"
    assert config["catalog"]["sha256"] == "351eda39efea3f5eb4d394b91956d1c08e7f63993c41d09d686e5ad2983cbbea"
    assert config["xrs_l2"]["version"] == "v2-2-1"
    assert config["input_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert len(lock["xrs_l2"]["files"]) == 43
    assert all(len(item["sha256"]) == 64 for item in lock["xrs_l2"]["files"])


def test_existing_event_output_is_a_frozen_regression_reference():
    import csv

    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/paper_background_sensitivity/event_backgrounds.csv"
    summary_path = root / "outputs/paper_background_sensitivity/summary.json"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "bd038f58d308f98c73e5f44176138c0baccce2cae5c19406dc639c466ee5bedf"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["cohort_events"] == 41_723
    assert summary["uncapped_background_valid"] == 41_723
    assert summary["capped_background_valid"] == 41_220
    assert summary["events_uncapped_rxfi_negative"] == 4_842
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 41_723
    assert [(row["time"], row["flare_id"]) for row in rows] == sorted(
        (row["time"], row["flare_id"]) for row in rows
    )


def test_l2_url_routes_legacy_and_goes_r_satellites_to_noaa_hosts():
    assert l2_url(14, 2010, "v2-2-1") == (
        "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/"
        "goes14/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g14_y2010_v2-2-1.nc"
    )
    assert l2_url(16, 2025, "v2-2-1") == (
        "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
        "goes16/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g16_y2025_v2-2-1.nc"
    )


def test_cache_path_is_namespaced_by_product_version(tmp_path):
    path = cache_path(tmp_path, 16, 2025, "v2-2-1")

    assert path == tmp_path / "xrs_l2/v2-2-1/g16/2025.nc"
    assert cache_path(tmp_path, 16, 2025, "v2-3-0") != path


def test_find_cached_input_accepts_legacy_layout_only_when_hash_is_pinned(tmp_path):
    payload = b"frozen legacy cache input"
    legacy = tmp_path / "xrs_l2/g16/2025.nc"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    path, record = find_cached_input(
        tmp_path, 16, 2025, "v2-2-1", expected_sha256=digest, expected_size=len(payload)
    )

    assert path == legacy
    assert record["legacy_layout"] is True
    assert record["sha256"] == digest


def test_find_cached_input_rejects_wrong_hash_in_existing_legacy_cache(tmp_path):
    legacy = tmp_path / "xrs_l2/g16/2025.nc"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"wrong product version")

    with pytest.raises(InputDownloadError, match="SHA-256 mismatch"):
        find_cached_input(tmp_path, 16, 2025, "v2-2-1", expected_sha256="0" * 64)


@pytest.fixture
def local_noaa_server(monkeypatch):
    payload = b"locked test payload"

    class Response:
        def __init__(self, body, content_type="application/octet-stream", interrupted=False):
            self.status = 200
            self.headers = {"Content-Type": content_type}
            self.body = body
            self.interrupted = interrupted
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size=-1):
            self.read_count += 1
            if self.interrupted and self.read_count == 1:
                return self.body
            if self.interrupted:
                raise http.client.IncompleteRead(self.body)
            if self.read_count == 1:
                return self.body
            return b""

    base = "https://noaa.test"

    def fake_urlopen(request, timeout=60):
        del timeout
        path = request.full_url.removeprefix(base)
        if path == "/missing":
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        if path == "/html":
            return Response(b"<html>not a data file</html>", "text/html")
        if path == "/interrupted":
            return Response(b"partial", interrupted=True)
        if path == "/empty":
            return Response(b"")
        return Response(payload)

    import urllib.error

    monkeypatch.setattr("paper_background.noaa.urllib.request.urlopen", fake_urlopen)
    return base, payload


def test_download_atomic_pins_size_hash_and_renames_complete_file(tmp_path, local_noaa_server):
    base, payload = local_noaa_server
    destination = tmp_path / "v2-2-1/g16/2025.nc"

    result = download_atomic(
        f"{base}/ok",
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    assert destination.read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["size_bytes"] == len(payload)
    assert not destination.with_suffix(destination.suffix + ".part").exists()


@pytest.mark.parametrize("route", ["missing", "html", "interrupted", "empty", "mismatch"])
def test_download_atomic_rejects_bad_or_incomplete_responses(tmp_path, local_noaa_server, route):
    base, _ = local_noaa_server
    destination = tmp_path / f"{route}.nc"

    with pytest.raises(InputDownloadError):
        download_atomic(
            f"{base}/{route}", destination,
            expected_sha256="0" * 64,
            expected_size=len(b"locked test payload") if route == "mismatch" else 100,
            timeout_seconds=3,
        )

    assert not destination.exists()
    assert not destination.with_suffix(destination.suffix + ".part").exists()


def test_read_l2_year_decodes_cf_time_and_returns_sorted_arrays(tmp_path):
    import xarray as xr

    timestamps = np.array(["2019-12-31T23:59:00", "2020-01-01T00:00:00"], dtype="datetime64[s]")
    source = xr.Dataset(
        {"xrsb_flux": ("time", [1.0e-7, 2.0e-7]), "xrsb_flag": ("time", [0, 1])},
        coords={"time": timestamps},
    )
    path = tmp_path / "tiny.nc"
    source.to_netcdf(path, engine="h5netcdf")

    seconds, flux, flags, metadata = read_l2_year(path)

    expected = np.array([
        int(datetime(2019, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp()),
        int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()),
    ])
    assert seconds.tolist() == expected.tolist()
    assert flux.tolist() == [1.0e-7, 2.0e-7]
    assert flags.tolist() == [0, 1]
    assert metadata["sample_count"] == 2


def test_read_l2_year_rejects_missing_required_variable(tmp_path):
    import xarray as xr

    source = xr.Dataset({"xrsb_flux": ("time", [1.0e-7])}, coords={"time": np.array(["2020-01-01"], dtype="datetime64[s]")})
    path = tmp_path / "invalid.nc"
    source.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="xrsb_flag"):
        read_l2_year(path)


def _make_runner_fixture(root: Path):
    import csv
    import xarray as xr

    from paper_background.noaa import cache_path

    cache_root = root / "cache"
    catalog_path = root / "fixture.csv"
    fields = [
        "time", "start_time", "end_time", "flare_id", "xrsb_irrad",
        "flare_class", "xrsb_irrad_source", "background_irrad",
    ]
    rows = [
        {"time":"2020-01-01 00:02:00", "start_time":"2020-01-01 00:00:00", "end_time":"2020-01-01 00:10:00", "flare_id":"target", "xrsb_irrad":"4e-7", "flare_class":"B4.0", "xrsb_irrad_source":"GOES-16", "background_irrad":"5e-8"},
        {"time":"2019-12-31 12:01:00", "start_time":"2019-12-31 12:00:00", "end_time":"2019-12-31 12:02:00", "flare_id":"other", "xrsb_irrad":"2e-7", "flare_class":"B2.0", "xrsb_irrad_source":"GOES-16", "background_irrad":"4e-8"},
    ]
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    years = {
        2019: (
            ["2019-12-31T00:00:00", "2019-12-31T12:00:00", "2019-12-31T12:01:00", "2019-12-31T12:02:00", "2019-12-31T23:59:00"],
            [1e-7, 7e-7, 8e-7, 9e-7, 3e-7],
        ),
        2020: (["2020-01-01T00:00:00", "2020-01-01T00:01:00"], [2e-7, 5e-7]),
    }
    lock_files = []
    for year, (times, values) in years.items():
        data_path = cache_path(cache_root, 16, year, "v2-2-1")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"xrsb_flux": ("time", values), "xrsb_flag": ("time", np.zeros(len(times), dtype=int))},
            coords={"time": np.asarray(times, dtype="datetime64[s]")},
        ).to_netcdf(data_path, engine="h5netcdf")
        data_bytes = data_path.read_bytes()
        lock_files.append({
            "satellite":16,"year":year,
            "filename":f"sci_xrsf-l2-avg1m_g16_y{year}_v2-2-1.nc",
            "url":f"https://data.ngdc.noaa.gov/fixture/g16/{year}.nc",
            "size_bytes":len(data_bytes),"sha256":hashlib.sha256(data_bytes).hexdigest(),
        })

    catalog_bytes = catalog_path.read_bytes()
    config_path = root / "config.json"
    lock_path = root / "input.lock.json"
    lock = {
        "lock_schema_version":1,
        "catalog":{"filename":catalog_path.name,"url":"https://noaa.test/fixture.csv","size_bytes":len(catalog_bytes),"sha256":hashlib.sha256(catalog_bytes).hexdigest(),"version":"v1-test"},
        "xrs_l2":{"product":"science-quality GOES XRS Level 2 avg1m","version":"v2-2-1","files":lock_files},
    }
    lock_path.write_text(json.dumps(lock,indent=2)+"\n")
    config = {
        "config_schema_version":1,"algorithm_version":"test-v1","output_schema_version":1,
        "catalog":{"filename":catalog_path.name,"url":"https://noaa.test/fixture.csv","sha256":lock["catalog"]["sha256"],"version":"v1-test","peak_time_start_utc":"2010-01-01T00:00:00Z","peak_time_end_exclusive_utc":"2026-01-01T00:00:00Z","deduplicate_key":["flare_id","time"]},
        "input_lock":"input.lock.json","input_lock_sha256":hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "xrs_l2":{"product":lock["xrs_l2"]["product"],"version":"v2-2-1","channel":"xrsb_flux","quality_flag":"xrsb_flag","valid_quality_value":0,"satellite_field":"xrsb_irrad_source","cache_subdirectory":"xrs_l2"},
        "method":{"window_seconds":86400,"window_start_inclusive":True,"window_end_exclusive":True,"other_flare_start_inclusive":True,"other_flare_end_inclusive":True,"valid_flux":"finite_positive","upper_cap":"xrsb_flux_at_exact_catalog_start_time_same_satellite","upper_cap_inclusive":True,"rxfi":"(peak_flux - background) / background","average":"arithmetic_mean"},
        "output":{"float_format":".12g","event_order":["time","flare_id"],"directory":"outputs/runs"},
    }
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    return config_path, catalog_path, cache_root


def test_offline_runner_reproduces_year_boundary_event_calculations(tmp_path):
    config_path, catalog_path, cache_root = _make_runner_fixture(tmp_path)

    output_a = run_calculation(
        config_path, catalog_path=catalog_path, cache_root=cache_root,
        output_root=tmp_path / "out-a", offline=True,
    )
    output_b = run_calculation(
        config_path, catalog_path=catalog_path, cache_root=cache_root,
        output_root=tmp_path / "out-b", offline=True,
    )

    assert output_a.name == output_b.name
    csv_a = (output_a / "event_backgrounds.csv").read_bytes()
    csv_b = (output_b / "event_backgrounds.csv").read_bytes()
    assert csv_a == csv_b
    rows = {row["flare_id"]: row for row in csv.DictReader(csv_a.decode().splitlines())}
    target = rows["target"]
    assert target["uncapped_retained_minutes"] == "2"
    assert target["other_flare_masked_minutes"] == "3"
    assert target["capped_retained_minutes"] == "1"
    assert target["background_uncapped_W_m2"] == "2e-07"
    assert target["background_capped_W_m2"] == "1e-07"
    assert target["rxfi_uncapped_fractional_excess"] == "1"
    assert target["rxfi_capped_fractional_excess"] == "3"


def test_offline_runner_fails_preflight_before_creating_output(tmp_path):
    config_path, catalog_path, _cache_root = _make_runner_fixture(tmp_path)
    missing_cache = tmp_path / "empty-cache"
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(FileNotFoundError, match="not cached"):
        run_calculation(
            config_path, catalog_path=catalog_path, cache_root=missing_cache,
            output_root=output_root, offline=True,
        )

    assert not output_root.exists()


def test_runner_refuses_to_overwrite_same_fingerprint_without_flag(tmp_path):
    config_path, catalog_path, cache_root = _make_runner_fixture(tmp_path)
    output_root = tmp_path / "runs"
    first = run_calculation(
        config_path, catalog_path=catalog_path, cache_root=cache_root,
        output_root=output_root, offline=True,
    )
    original = (first / "event_backgrounds.csv").read_bytes()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_calculation(
            config_path, catalog_path=catalog_path, cache_root=cache_root,
            output_root=output_root, offline=True,
        )

    assert (first / "event_backgrounds.csv").read_bytes() == original
