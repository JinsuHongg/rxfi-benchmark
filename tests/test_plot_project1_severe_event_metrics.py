"""Regression checks for the Project 1 severe-event proposal figures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_project1_severe_event_metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_project1_severe_event_metrics", SCRIPT)
    assert spec and spec.loader, "proposal plotting script must be importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loads_exact_mplus_test_metrics_and_validation_frozen_metadata() -> None:
    """The main slide data must remain tied to saved M+ TEST output only."""
    assert SCRIPT.exists(), "proposal plotting script is missing"
    module = load_module()

    metrics, prevalence = module.load_event_metrics(ROOT / "outputs" / "severe_event_binary_metrics" / "project1_qr_test_summary.csv", "M+")

    assert metrics["target"].tolist() == [
        "max_peak_flux",
        "cumulative_peak_flux",
        "max_rxfi",
        "cumulative_rxfi",
    ]
    assert metrics["evaluation_split"].unique().tolist() == ["test"]
    assert metrics["threshold_selection_split"].unique().tolist() == ["validation"]
    assert metrics["threshold_rule"].str.startswith("maximize_validation_").all()
    assert prevalence == pytest.approx(8596 / 27980)
    assert metrics.loc[metrics["target"] == "cumulative_peak_flux", "AUPRC"].item() == pytest.approx(0.7147065875468492)
    assert metrics.loc[metrics["target"] == "cumulative_peak_flux", "TSS"].item() == pytest.approx(0.5945926293449217)
    assert metrics.loc[metrics["target"] == "cumulative_peak_flux", "HSS"].item() == pytest.approx(0.5006279861276501)
    assert metrics.loc[metrics["target"] == "cumulative_peak_flux", "F1_positive"].item() == pytest.approx(0.6927359721618095)


def test_creates_requested_mplus_and_xplus_export_formats(tmp_path: Path) -> None:
    """Both slide-ready figures export cleanly in PNG, PDF, and SVG formats."""
    assert SCRIPT.exists(), "proposal plotting script is missing"
    module = load_module()

    module.main(["--output-dir", str(tmp_path)])

    for stem in ("project1_mplus_metric_comparison", "project1_xplus_metric_comparison"):
        for suffix in (".png", ".pdf", ".svg"):
            output = tmp_path / f"{stem}{suffix}"
            assert output.is_file()
            assert output.stat().st_size > 0


def test_rejects_metrics_without_saved_validation_threshold_provenance(tmp_path: Path) -> None:
    """The figure must not claim validation-frozen thresholds without saved evidence."""
    assert SCRIPT.exists(), "proposal plotting script is missing"
    module = load_module()
    source = ROOT / "outputs" / "severe_event_binary_metrics" / "project1_qr_test_summary.csv"
    incomplete = pd.read_csv(source).drop(columns=["selected_threshold"])
    incomplete_path = tmp_path / "incomplete_summary.csv"
    incomplete.to_csv(incomplete_path, index=False)

    with pytest.raises(ValueError, match="selected_threshold"):
        module.load_event_metrics(incomplete_path, "M+")
