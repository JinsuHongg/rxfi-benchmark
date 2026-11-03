#!/usr/bin/env python3
"""Evaluate one saved XRS quantile-regression checkpoint on test data."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

from scripts.training.qr_lightning import QuantileRegressionModule
from scripts.training.train_vit_quantile_regression import (
    PREDICTION_FIELDS,
    SuryaDataModule,
    prediction_rows,
    resolve_target_transform,
    validate_config,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint, config, output, and W&B options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--task-index", default=os.environ.get("SLURM_ARRAY_TASK_ID", "local"))
    parser.add_argument("--precision", choices=("32", "16-mixed", "bf16-mixed"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def prediction_filename(
    target: str, seed: int, job_id: str, task_index: str
) -> str:
    """Create a flat CSV filename unique to a target and Slurm task."""
    safe_target = re.sub(r"[^A-Za-z0-9_-]", "_", target)
    safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)
    safe_task_index = re.sub(r"[^A-Za-z0-9_-]", "_", task_index)
    return (
        f"{safe_target}_seed{seed}_job{safe_job_id}_task{safe_task_index}"
        "_test_predictions.csv"
    )


def slurm_job_id() -> str:
    """Return an array-level job ID or a local fallback label."""
    return os.environ.get(
        "SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local")
    )


def load_run_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen training config and its train-fitted transform."""
    loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping in config {args.config}")
    config = cast(dict[str, Any], loaded)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.precision is not None:
        config["precision"] = args.precision
    config["target_transform"] = resolve_target_transform(config)
    spec = validate_config(config)
    if config["model_name"] != "compact_cnn_v1":
        raise ValueError("Final XRS test evaluation requires compact_cnn_v1")
    if config["target_column"] != args.expected_target:
        raise ValueError("Manifest target does not match the resolved config")
    if int(config["seed"]) != args.expected_seed:
        raise ValueError("Manifest seed does not match the resolved config")
    if args.checkpoint.resolve().is_file() is False:
        raise FileNotFoundError(args.checkpoint)
    return config, spec


def build_run_name(config: dict[str, Any]) -> str:
    """Build a distinct UTC W&B name for one target and seed."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
    learning_rate = f"{float(config['learning_rate']):.0e}"
    learning_rate = learning_rate.replace("e-0", "e-").replace("e+0", "e+")
    weight_decay = f"{float(config['weight_decay']):g}"
    return (
        f"{timestamp}-test-{config['model_name']}-{config['target_column']}"
        f"-lr{learning_rate}-wd{weight_decay}-seed{config['seed']}"
    )


def checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Read training-selection metadata stored beside a best checkpoint."""
    metadata_path = checkpoint_path.parent.parent / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "validation":
        raise ValueError("Checkpoint was not selected using validation data")
    if Path(payload.get("best_checkpoint_path", "")).resolve() != checkpoint_path.resolve():
        raise ValueError("Checkpoint metadata points to a different checkpoint")
    return cast(dict[str, Any], payload)


def make_wandb_logger(
    args: argparse.Namespace,
    config: dict[str, Any],
    data: SuryaDataModule,
    run_metadata: dict[str, Any],
) -> WandbLogger:
    """Create a W&B run containing config and test cohort provenance."""
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError("WANDB_ENTITY must be set")
    logged_config = {
        **config,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_selection_split": run_metadata["selection_split"],
        "best_validation_epoch": run_metadata["best_validation_epoch"],
        "test_cohort": data.cohort["test"],
        "test_split_csv": str(
            Path(config["target_split_dir"]) / "test.csv"
        ),
        "test_prediction_export_enabled": True,
        "test_prediction_csv": str(
            args.output_root
            / "predictions"
            / prediction_filename(
                str(config["target_column"]),
                int(config["seed"]),
                slurm_job_id(),
                str(args.task_index),
            )
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": args.task_index,
    }
    return WandbLogger(
        entity=entity,
        project=args.wandb_project,
        name=build_run_name(config),
        group=args.wandb_group,
        job_type="final-test-evaluation",
        config=logged_config,
        save_dir=str(args.output_root),
        log_model=False,
        tags=["xrs-only", "compact-cnn-v1", "final-test", config["target_column"]],
    )


def trainer_precision(config: dict[str, Any]) -> str:
    """Map the project precision setting to Lightning's explicit form."""
    return {
        "32": "32-true",
        "16-mixed": "16-mixed",
        "bf16-mixed": "bf16-mixed",
    }[config["precision"]]


def evaluate_checkpoint(
    args: argparse.Namespace,
    config: dict[str, Any],
    spec: dict[str, Any],
    data: SuryaDataModule,
    run_metadata: dict[str, Any],
) -> None:
    """Run test metrics and sample predictions from the selected checkpoint."""
    args.output_root.mkdir(parents=True, exist_ok=True)
    wandb_logger = make_wandb_logger(args, config, data, run_metadata)
    model = QuantileRegressionModule(config)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=cast(Any, trainer_precision(config)),
        logger=wandb_logger,
        enable_checkpointing=False,
        default_root_dir=str(args.output_root),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    try:
        test_results = trainer.test(
            model,
            dataloaders=data.test_dataloader(),
            ckpt_path=str(args.checkpoint.resolve()),
            verbose=True,
        )
        save_test_artifacts(
            model,
            data,
            config,
            spec,
            args,
            run_metadata,
            test_results,
        )
    finally:
        wandb_logger.experiment.finish()


def save_test_artifacts(
    model: QuantileRegressionModule,
    data: SuryaDataModule,
    config: dict[str, Any],
    spec: dict[str, Any],
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
    test_results: Sequence[Mapping[str, float]],
) -> None:
    """Write uniquely named per-sample predictions and scalar metrics."""
    if len(test_results) != 1:
        raise RuntimeError("Expected exactly one test metric result")
    predictions = model.test_predictions
    if not predictions:
        raise RuntimeError("Test evaluation returned no sample predictions")
    filename = prediction_filename(
        str(config["target_column"]),
        int(config["seed"]),
        slurm_job_id(),
        str(args.task_index),
    )
    prediction_path = args.output_root / "predictions" / filename
    rows = prediction_rows(
        predictions,
        data.records["test"],
        config,
        spec,
    )
    write_csv(prediction_path, rows, PREDICTION_FIELDS)
    save_test_metrics(
        args, config, run_metadata, test_results[0], len(rows), filename
    )
    logger.info("Saved {} test predictions to {}", len(rows), prediction_path)


def save_test_metrics(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
    metrics: Mapping[str, float],
    sample_count: int,
    prediction_file: str,
) -> None:
    """Store one row of target, checkpoint, cohort, and test metric data."""
    metric_values = {
        key.removeprefix("test/"): float(value)
        for key, value in metrics.items()
    }
    job_id = slurm_job_id()
    task_id = str(args.task_index)
    output_name = prediction_file.replace("_test_predictions.csv", "_test_metrics.csv")
    row = {
        "target": config["target_column"],
        "model": config["model_name"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_selection_split": run_metadata["selection_split"],
        "best_validation_epoch": run_metadata["best_validation_epoch"],
        "test_sample_count": sample_count,
        "prediction_file": prediction_file,
        "slurm_job_id": job_id,
        "slurm_task_id": task_id,
        **metric_values,
    }
    metric_path = args.output_root / "metrics" / output_name
    write_csv(metric_path, [row])
    logger.info("Saved test metrics to {}", metric_path)


def main() -> None:
    """Run one checkpoint-only test evaluation and local exports."""
    args = parse_args()
    config, spec = load_run_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing CPU fallback")
    run_metadata = checkpoint_metadata(args.checkpoint)
    data = SuryaDataModule(config, spec)
    data.setup("test")
    logger.info("Prepared test cohort: {}", data.cohort["test"])
    evaluate_checkpoint(args, config, spec, data, run_metadata)


if __name__ == "__main__":
    main()
