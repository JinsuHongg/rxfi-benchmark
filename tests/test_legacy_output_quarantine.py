"""Ensure superseded 224x224 experiment outputs stay quarantined."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "outputs" / "legacy_224x224"


def test_224x224_model_outputs_are_quarantined() -> None:
    expected = (
        "four_qr_target_analysis",
        "project2",
        "project2_channel_audit.json",
        "project3_exports",
        "severe_event_binary_metrics",
    )
    assert (LEGACY / "README.md").is_file()
    assert all((LEGACY / name).exists() for name in expected)
    assert all(not (ROOT / "outputs" / name).exists() for name in expected)


def test_legacy_project3_exports_are_marked_as_224x224() -> None:
    import json

    metadata = LEGACY / "project3_exports" / "max_peak_flux" / "metadata.json"
    assert metadata.is_file()
    assert json.loads(metadata.read_text())["image_configuration"]["image_size"] == 224
