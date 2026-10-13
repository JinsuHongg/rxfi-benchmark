"""Structural checks for the topic-organized script tree."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_MOVES = {
    "compute_paper_background_sensitivity.py": "background/compute_paper_background_sensitivity.py",
    "paper_background/__init__.py": "background/paper_background/__init__.py",
    "paper_background/calculation.py": "background/paper_background/calculation.py",
    "paper_background/noaa.py": "background/paper_background/noaa.py",
    "paper_background/runner.py": "background/paper_background/runner.py",
    "generate_24h_targets.py": "data/generate_24h_targets.py",
    "inspect_noaa_xrs_flare_report.py": "data/inspect_noaa_xrs_flare_report.py",
    "refine_no_flare_max_peak_flux.py": "data/refine_no_flare_max_peak_flux.py",
    "analyze_four_qr_targets.py": "analysis/analyze_four_qr_targets.py",
    "analyze_rxfi_distribution.py": "analysis/analyze_rxfi_distribution.py",
    "audit_fq_xrs_threshold_exceedance.py": "analysis/audit_fq_xrs_threshold_exceedance.py",
    "audit_historical_goes_label_mismatch.py": "analysis/audit_historical_goes_label_mismatch.py",
    "evaluate_severe_event_binary_metrics.py": "analysis/evaluate_severe_event_binary_metrics.py",
    "severe_event_metrics.py": "analysis/severe_event_metrics.py",
    "project2/analyze_project2_preliminary.py": "analysis/project2/analyze_project2_preliminary.py",
    "project2/audit_project2_channels.py": "analysis/project2/audit_project2_channels.py",
    "plot_four_24h_targets_noaa.py": "visualization/plot_four_24h_targets_noaa.py",
    "plot_four_24h_targets_xrs_1min.py": "visualization/plot_four_24h_targets_xrs_1min.py",
    "plot_project1_severe_event_metrics.py": "visualization/plot_project1_severe_event_metrics.py",
    "plot_single_flare_classification.py": "visualization/plot_single_flare_classification.py",
    "benchmark_qr_batch_size.py": "training/benchmark_qr_batch_size.py",
    "train_vit_classifier.py": "training/train_vit_classifier.py",
    "train_vit_quantile_regression.py": "training/train_vit_quantile_regression.py",
    "run_all_qr_experiments.sh": "launchers/run_all_qr_experiments.sh",
    "slurm_vit_small_224_max_flare_class.sbatch": "launchers/slurm_vit_small_224_max_flare_class.sbatch",
    "slurm/run_qr_array.sbatch": "launchers/slurm/run_qr_array.sbatch",
    "project2/run_project2_preliminary.sh": "launchers/project2/run_project2_preliminary.sh",
    "project2/slurm/run_project2_ordinal6_classification_array.sbatch": "launchers/project2/slurm/run_project2_ordinal6_classification_array.sbatch",
    "project2/slurm/run_project2_preliminary_array.sbatch": "launchers/project2/slurm/run_project2_preliminary_array.sbatch",
    "export_project3_predictions.py": "exports/project3/export_project3_predictions.py",
    "project3_export_contract.py": "exports/project3/project3_export_contract.py",
    "project3_export_loader_example.py": "exports/project3/project3_export_loader_example.py",
    "validate_project3_exports.py": "exports/project3/validate_project3_exports.py",
}


def test_all_script_files_are_moved_once_to_the_approved_topic_paths():
    assert len(SCRIPT_MOVES) == 33
    for old_relative, new_relative in SCRIPT_MOVES.items():
        assert (SCRIPT_ROOT / new_relative).is_file(), new_relative
        assert not (SCRIPT_ROOT / old_relative).exists(), old_relative


def test_colocated_packages_import_at_their_topic_paths():
    background_path = str(SCRIPT_ROOT / "background")
    if background_path not in sys.path:
        sys.path.insert(0, background_path)
    modules = [
        "scripts.background.compute_paper_background_sensitivity",
        "scripts.background.paper_background.runner",
        "scripts.analysis.severe_event_metrics",
        "scripts.analysis.evaluate_severe_event_binary_metrics",
        "scripts.exports.project3.project3_export_contract",
    ]
    for module in modules:
        importlib.import_module(module)


def test_nested_project_modules_resolve_repository_root():
    project2 = importlib.import_module("scripts.analysis.project2.analyze_project2_preliminary")
    project3 = importlib.import_module("scripts.exports.project3.export_project3_predictions")

    assert project2.ROOT.resolve() == ROOT
    assert project3.ROOT.resolve() == ROOT


def test_moved_shell_and_slurm_files_keep_their_executable_mode():
    for old_relative, new_relative in SCRIPT_MOVES.items():
        if not old_relative.endswith((".sh", ".sbatch")):
            continue
        entry = subprocess.run(
            ["git", "ls-tree", "HEAD", f"scripts/{old_relative}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert entry, f"no Git baseline for scripts/{old_relative}"
        baseline_mode = entry.split()[0]
        moved_mode = "100755" if (SCRIPT_ROOT / new_relative).stat().st_mode & 0o111 else "100644"
        assert moved_mode == baseline_mode, new_relative


def test_every_script_directory_has_a_readme_that_lists_its_direct_files():
    directories = [SCRIPT_ROOT, *(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())]
    directories = [path for path in directories if "__pycache__" not in path.parts]
    for directory in directories:
        readme = directory / "README.md"
        assert readme.is_file(), f"missing {readme.relative_to(ROOT)}"
        content = readme.read_text(encoding="utf-8")
        for child in directory.iterdir():
            if child.is_file() and child.name != "README.md" and "__pycache__" not in child.parts:
                assert child.name in content, f"{child.name} is not documented in {readme.relative_to(ROOT)}"
