"""Lightning modules for Zarr-backed quantile regression."""
from __future__ import annotations

import time
from typing import Any

import lightning.pytorch as pl
import timm
import torch
from loguru import logger
from torchmetrics import MeanSquaredError
from transformers import get_cosine_schedule_with_warmup


class QuantileRegressionModule(pl.LightningModule):
    """ViT-Small quantile model, metrics, optimizer, and learning-rate policy."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Build the configured ViT and retain optimizer/scheduler settings."""
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
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
                split: MeanSquaredError(squared=False)
                for split in ("train", "validation", "test")
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one predicted value per configured quantile."""
        return self.model(images)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], split: str
    ) -> torch.Tensor:
        """Compute pinball loss and batch-weighted aggregate diagnostics."""
        images, target = batch
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
        rmse_q50 = self.rmse_q50_metrics[split]
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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one optimization batch and log weighted training metrics."""
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one validation batch for checkpoint selection."""
        del batch_idx
        return self._shared_step(batch, "validation")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Evaluate one test batch after training is complete."""
        del batch_idx
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Return CPU predictions while preserving loader row order."""
        del batch_idx
        images, _ = batch
        return self(images).float().cpu()

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
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
        for split in ("train", "validation"):
            for key, value in values.items():
                if key.startswith(f"{split}/"):
                    name = key.removeprefix(f"{split}/")
                    row[f"{split}_{name}"] = float(value.detach().cpu())
        if "validation_mse_q50" in row:
            row["validation_rmse_q50"] = float(row["validation_mse_q50"]) ** 0.5
        self.rows.append(row)
