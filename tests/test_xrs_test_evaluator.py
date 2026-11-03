"""Behavioral checks for checkpoint-only XRS test evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.evaluate_xrs_only_qr import prediction_filename
from scripts.training.qr_lightning import QuantileRegressionModule


class FixedQuantiles(torch.nn.Module):
    """Return ordered quantiles derived from each input value."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Build three ordered predictions from the one-dimensional feature."""
        median = images[:, 0]
        return torch.stack((median - 0.5, median, median + 0.5), dim=1)


def test_prediction_filename_is_unique_for_target_seed_and_array_task() -> None:
    """Keep flat prediction exports distinct across concurrent array tasks."""
    assert prediction_filename("max_xflux", 2, "4812345", "7") == (
        "max_xflux_seed2_job4812345_task7_test_predictions.csv"
    )


def test_test_stage_logs_validation_equivalent_overall_metrics() -> None:
    """Expose rank, fit, calibration, raw-error, and quantile test metrics."""
    config: dict[str, Any] = {
        "model_name": "compact_cnn_v1",
        "in_chans": 1,
        "quantiles": [0.05, 0.5, 0.95],
        "cnn_channels": [2],
        "cnn_stem_kernel_size": 3,
        "cnn_dropout": 0.0,
        "target_transform": {
            "name": "asinh_zscore",
            "scale_s": 1.0,
            "transformed_mean": 0.0,
            "transformed_std_population": 1.0,
        },
    }
    model = QuantileRegressionModule(config)
    model.model = FixedQuantiles()
    feature = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    target = feature[:, 0] + 0.25
    periods = torch.full((4,), -1, dtype=torch.int64)
    loader = DataLoader(TensorDataset(feature, target, periods), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    results = trainer.test(model, dataloaders=loader, verbose=False)

    assert results
    metrics: Mapping[str, float] = results[0]
    expected = {
        "test/pinball_loss",
        "test/pinball_q05",
        "test/pinball_q50",
        "test/pinball_q95",
        "test/fraction_target_le_q05",
        "test/fraction_target_le_q50",
        "test/fraction_target_le_q95",
        "test/rmse_q50",
        "test/mae_q50",
        "test/mse_q50",
        "test/spearman_q50",
        "test/r2_q50_transformed",
        "test/interval_coverage_90",
        "test/mean_interval_width_90_transformed",
        "test/mae_q50_raw",
        "test/cross_q05_gt_q50",
        "test/cross_q50_gt_q95",
        "test/cross_q05_gt_q95",
    }
    assert expected <= metrics.keys()
    assert metrics["test/spearman_q50"] == pytest.approx(1.0)
    assert metrics["test/r2_q50_transformed"] == pytest.approx(0.95)
