"""Lightning modules for Zarr-backed quantile regression."""
from __future__ import annotations

import time
from typing import Any, cast

import lightning.pytorch as pl
import timm
import torch
from loguru import logger
from torchmetrics import MeanSquaredError, Metric, R2Score, SpearmanCorrCoef
from transformers import get_cosine_schedule_with_warmup


class CompactCNN(torch.nn.Module):
    """Small convolutional regressor with a broad initial receptive field."""

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        channels: list[int],
        stem_kernel_size: int,
        dropout: float,
    ) -> None:
        """Build convolution stages followed by pooled quantile predictions.

        Args:
            in_channels: Number of image channels.
            output_channels: Number of predicted target quantiles.
            channels: Output width for each downsampling stage.
            stem_kernel_size: Kernel width used by the first convolution.
            dropout: Dropout probability before the prediction head.
        """
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_channels = in_channels
        for index, output_width in enumerate(channels):
            kernel_size = stem_kernel_size if index == 0 else 3
            layers.extend(
                [
                    torch.nn.Conv2d(
                        current_channels,
                        output_width,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(output_width),
                    torch.nn.GELU(),
                    torch.nn.Conv2d(
                        output_width,
                        output_width,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(output_width),
                    torch.nn.GELU(),
                ]
            )
            current_channels = output_width
        self.features = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(current_channels, output_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of channel-first images to target quantiles."""
        features = self.features(images)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.head(self.dropout(pooled))


class QuantileRegressionModule(pl.LightningModule):
    """Quantile model, metrics, optimizer, and learning-rate policy."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Build the configured model and retain optimizer/scheduler settings."""
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.scheduler_metadata: dict[str, Any] = {}
        if config["model_name"] == "compact_cnn_v1":
            self.model = CompactCNN(
                in_channels=int(config["in_chans"]),
                output_channels=len(config["quantiles"]),
                channels=[int(value) for value in config["cnn_channels"]],
                stem_kernel_size=int(config["cnn_stem_kernel_size"]),
                dropout=float(config["cnn_dropout"]),
            )
        else:
            self.model = timm.create_model(
                config["model_name"],
                pretrained=False,
                in_chans=int(config["in_chans"]),
                num_classes=len(config["quantiles"]),
            )
        self.register_buffer(
            "quantiles", torch.tensor(config["quantiles"], dtype=torch.float32)
        )
        self.rmse_q50_metrics = torch.nn.ModuleDict(
            {
                f"{split}_split": MeanSquaredError(squared=False)
                for split in ("train", "validation", "test")
            }
        )
        self.validation_spearman_q50 = SpearmanCorrCoef()
        self.validation_r2_q50 = R2Score()
        self.validation_period_names = list(config.get("validation_periods", {}))
        self.period_spearman_q50 = torch.nn.ModuleDict(
            {name: SpearmanCorrCoef() for name in self.validation_period_names}
        )
        self.period_r2_q50 = torch.nn.ModuleDict(
            {name: R2Score() for name in self.validation_period_names}
        )
        self.period_rmse_q50 = torch.nn.ModuleDict(
            {
                name: MeanSquaredError(squared=False)
                for name in self.validation_period_names
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one predicted value per configured quantile."""
        return self.model(images)

    def inverse_target(self, values: torch.Tensor) -> torch.Tensor:
        """Map standardized asinh targets back to physical units."""
        transform = self.config["target_transform"]
        return float(transform["scale_s"]) * torch.sinh(
            values * float(transform["transformed_std_population"])
            + float(transform["transformed_mean"])
        )

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], split: str
    ) -> torch.Tensor:
        """Compute pinball loss and batch-weighted aggregate diagnostics."""
        images, target, period_indices = batch
        prediction = self(images)
        quantiles = self.quantiles.to(dtype=prediction.dtype)
        error = target[:, None] - prediction
        pinball = torch.maximum(quantiles * error, (quantiles - 1) * error)
        loss = pinball.mean()
        batch_size = target.shape[0]
        self.log(
            f"{split}/pinball_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=split == "validation",
            sync_dist=True,
        )
        per_quantile = pinball.mean(dim=0)
        for index, quantile in enumerate(self.config["quantiles"]):
            suffix = int(float(quantile) * 100)
            self.log(
                f"{split}/pinball_q{suffix:02d}",
                per_quantile[index],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            coverage = (target <= prediction[:, index]).float().mean()
            self.log(
                f"{split}/fraction_target_le_q{suffix:02d}",
                coverage,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        median_error = target - prediction[:, 1]
        rmse_q50 = cast(Metric, self.rmse_q50_metrics[f"{split}_split"])
        rmse_q50.update(prediction[:, 1], target)
        self.log(
            f"{split}/rmse_q50",
            rmse_q50,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        metrics = {
            "mae_q50": median_error.abs().mean(),
            "mse_q50": median_error.square().mean(),
            "cross_q05_gt_q50": (prediction[:, 0] > prediction[:, 1])
            .float()
            .mean(),
            "cross_q50_gt_q95": (prediction[:, 1] > prediction[:, 2])
            .float()
            .mean(),
            "cross_q05_gt_q95": (prediction[:, 0] > prediction[:, 2])
            .float()
            .mean(),
        }
        if split == "validation":
            self._log_validation_metrics(prediction, target, batch_size)
            self._log_validation_period_metrics(
                prediction, target, period_indices
            )
        for name, value in metrics.items():
            self.log(
                f"{split}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        if split == "train":
            self.log(
                "train/learning_rate",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
        return loss

    def _log_validation_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor, batch_size: int
    ) -> None:
        """Log rank, variance, interval, and raw-unit validation metrics."""
        self.validation_spearman_q50.update(prediction[:, 1], target)
        self.validation_r2_q50.update(prediction[:, 1], target)
        self.log(
            "validation/spearman_q50",
            self.validation_spearman_q50,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/r2_q50_transformed",
            self.validation_r2_q50,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        target_raw = self.inverse_target(target)
        median_raw = self.inverse_target(prediction[:, 1])
        metrics = {
            "validation/interval_coverage_90": (
                (target >= prediction[:, 0]) & (target <= prediction[:, 2])
            ).float().mean(),
            "validation/mean_interval_width_90_transformed": (
                prediction[:, 2] - prediction[:, 0]
            ).mean(),
            "validation/mae_q50_raw": (target_raw - median_raw).abs().mean(),
        }
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _log_validation_period_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        period_indices: torch.Tensor,
    ) -> None:
        """Log validation diagnostics separately for configured time windows."""
        for index, name in enumerate(self.validation_period_names):
            mask = period_indices == index
            if not bool(mask.any()):
                continue
            period_prediction = prediction[mask]
            period_target = target[mask]
            period_size = int(mask.sum())
            prefix = f"validation_{name}"
            error = period_target - period_prediction[:, 1]
            quantiles = self.quantiles.to(dtype=period_prediction.dtype)
            residual = period_target[:, None] - period_prediction
            pinball = torch.maximum(quantiles * residual, (quantiles - 1) * residual)
            self.period_spearman_q50[name].update(
                period_prediction[:, 1], period_target
            )
            self.period_r2_q50[name].update(period_prediction[:, 1], period_target)
            self.period_rmse_q50[name].update(
                period_prediction[:, 1], period_target
            )
            metrics = {
                f"{prefix}/pinball_loss": pinball.mean(),
                f"{prefix}/mae_q50": error.abs().mean(),
                f"{prefix}/interval_coverage_90": (
                    (period_target >= period_prediction[:, 0])
                    & (period_target <= period_prediction[:, 2])
                ).float().mean(),
                f"{prefix}/mean_interval_width_90_transformed": (
                    period_prediction[:, 2] - period_prediction[:, 0]
                ).mean(),
                f"{prefix}/mae_q50_raw": (
                    self.inverse_target(period_target)
                    - self.inverse_target(period_prediction[:, 1])
                ).abs().mean(),
            }
            for metric_name, value in metrics.items():
                self.log(
                    metric_name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=period_size,
                    sync_dist=True,
                )
            stateful_metrics = {
                "rmse_q50": cast(Metric, self.period_rmse_q50[name]),
                "spearman_q50": cast(Metric, self.period_spearman_q50[name]),
                "r2_q50_transformed": cast(Metric, self.period_r2_q50[name]),
            }
            for metric_name, metric in stateful_metrics.items():
                self.log(
                    f"{prefix}/{metric_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one optimization batch and log weighted training metrics."""
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one validation batch for checkpoint selection."""
        del batch_idx
        return self._shared_step(batch, "validation")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Evaluate one test batch after training is complete."""
        del batch_idx
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Return CPU predictions while preserving loader row order."""
        del batch_idx
        images, _, _ = batch
        return self(images).float().cpu()

    def configure_optimizers(self) -> Any:
        """Configure AdamW and an optional per-update cosine warmup schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        if self.config["lr_policy"] == "constant":
            return optimizer

        steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(
            1, round(float(self.config["warmup_ratio"]) * steps)
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps,
        )
        self.scheduler_metadata = {
            "total_optimizer_steps": steps,
            "warmup_optimizer_steps": warmup_steps,
            "warmup_ratio": float(self.config["warmup_ratio"]),
            "scheduler": "transformers.get_cosine_schedule_with_warmup",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class WarmupAwareEarlyStopping(pl.Callback):
    """Stop when validation loss stalls after the configured warmup period."""

    def __init__(self, monitor: str, patience: int, start_epoch: int) -> None:
        """Track validation improvements beginning at a one-based epoch."""
        super().__init__()
        if patience <= 0 or start_epoch <= 0:
            raise ValueError("patience and start_epoch must be positive")
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_score = float("inf")
        self.wait_count = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Count non-improving validation epochs after warmup completes."""
        del pl_module
        if trainer.current_epoch + 1 < self.start_epoch:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        score = float(metric.detach().cpu())
        if score < self.best_score:
            self.best_score = score
            self.wait_count = 0
            return
        self.wait_count += 1
        if self.wait_count >= self.patience:
            trainer.should_stop = True
            logger.info(
                "Early stopping at epoch {}: {} did not improve for {} epochs",
                trainer.current_epoch + 1,
                self.monitor,
                self.patience,
            )


class TrainingHistoryCallback(pl.Callback):
    """Collect epoch-level aggregate metrics for local CSV and curve exports."""

    def __init__(self) -> None:
        """Initialize an empty history for one fit run."""
        self.rows: list[dict[str, int | float]] = []
        self.epoch_started_at = 0.0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record the local start time used for the epoch duration field."""
        del trainer, pl_module
        self.epoch_started_at = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Attach finalized train metrics to the validation row for this epoch."""
        del pl_module
        train_metrics: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train/"):
                name = key.removeprefix("train/")
                train_metrics[f"train_{name}"] = float(value.detach().cpu())
        if self.rows and self.rows[-1]["epoch"] == trainer.current_epoch + 1:
            self.rows[-1].update(train_metrics)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Append completed train and validation aggregate metrics."""
        del pl_module
        values = trainer.callback_metrics
        row: dict[str, int | float] = {
            "epoch": int(trainer.current_epoch + 1),
            "elapsed_seconds": round(time.perf_counter() - self.epoch_started_at, 2),
        }
        for key, value in values.items():
            if key.startswith(("validation/", "validation_")):
                prefix, _, name = key.partition("/")
                row[f"{prefix}_{name}"] = float(value.detach().cpu())
        if "validation_mse_q50" in row:
            row["validation_rmse_q50"] = float(row["validation_mse_q50"]) ** 0.5
        self.rows.append(row)
