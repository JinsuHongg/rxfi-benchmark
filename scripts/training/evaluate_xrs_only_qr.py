#!/usr/bin/env python3
"""Evaluate one saved XRS quantile-regression checkpoint on a frozen split."""
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
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--task-index", default=os.environ.get("SLURM_ARRAY_TASK_ID", "local"))
    parser.add_argument("--precision", choices=("32", "16-mixed", "bf16-mixed"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def prediction_filename(
    target: str, seed: int, job_id: str, task_index: str, split: str
) -> str:
    """Create a flat CSV filename unique to a target and Slurm task."""
    safe_target = re.sub(r"[^A-Za-z0-9_-]", "_", target)
    safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)
    safe_task_index = re.sub(r"[^A-Za-z0-9_-]", "_", task_index)
    return (
        f"{safe_target}_seed{seed}_job{safe_job_id}_task{safe_task_index}"
        f"_{split}_predictions.csv"
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


def build_run_name(config: dict[str, Any], split: str) -> str:
    """Build a distinct UTC W&B name for one target and seed."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
    learning_rate = f"{float(config['learning_rate']):.0e}"
    learning_rate = learning_rate.replace("e-0", "e-").replace("e+0", "e+")
    weight_decay = f"{float(config['weight_decay']):g}"
    return (
        f"{timestamp}-{split}-{config['model_name']}-{config['target_column']}"
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
    split: str,
) -> WandbLogger:
    """Create a W&B run containing config and split cohort provenance."""
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError("WANDB_ENTITY must be set")
    logged_config = {
        **config,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_selection_split": run_metadata["selection_split"],
        "best_validation_epoch": run_metadata["best_validation_epoch"],
        f"{split}_cohort": data.cohort[split],
        f"{split}_split_csv": str(
            Path(config["target_split_dir"]) / f"{split}.csv"
        ),
        f"{split}_prediction_export_enabled": True,
        f"{split}_prediction_csv": str(
            args.output_root
            / "predictions"
            / prediction_filename(
                str(config["target_column"]),
                int(config["seed"]),
                slurm_job_id(),
                str(args.task_index),
                split,
            )
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": args.task_index,
    }
    return WandbLogger(
        entity=entity,
        project=args.wandb_project,
        name=build_run_name(config, split),
        group=args.wandb_group,
        job_type=f"{split}-checkpoint-evaluation",
        config=logged_config,
        save_dir=str(args.output_root),
        log_model=False,
        tags=["xrs-only", "compact-cnn-v1", split, config["target_column"]],
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
    """Run split metrics and sample predictions from the selected checkpoint."""
    split = args.split
    args.output_root.mkdir(parents=True, exist_ok=True)
    wandb_logger = make_wandb_logger(args, config, data, run_metadata, split)
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
        if split == "validation":
            model.validation_predictions.clear()
            model.collect_validation_predictions = True
            results = trainer.validate(
                model,
                dataloaders=data.val_dataloader(),
                ckpt_path=str(args.checkpoint.resolve()),
                verbose=True,
            )
            predictions = model.validation_predictions
        else:
            results = trainer.test(
                model,
                dataloaders=data.test_dataloader(),
                ckpt_path=str(args.checkpoint.resolve()),
                verbose=True,
            )
            predictions = model.test_predictions
        save_evaluation_artifacts(
            data,
            config,
            spec,
            args,
            run_metadata,
            results,
            predictions,
        )
    finally:
        wandb_logger.experiment.finish()


def save_evaluation_artifacts(
    data: SuryaDataModule,
    config: dict[str, Any],
    spec: dict[str, Any],
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
    evaluation_results: Sequence[Mapping[str, float]] | None,
    predictions: list[torch.Tensor],
) -> None:
    """Write uniquely named split predictions and scalar metrics."""
    if evaluation_results is None or len(evaluation_results) != 1:
        raise RuntimeError("Expected exactly one split metric result")
    if not predictions:
        raise RuntimeError("Split evaluation returned no sample predictions")
    split = args.split
    filename = prediction_filename(
        str(config["target_column"]),
        int(config["seed"]),
        slurm_job_id(),
        str(args.task_index),
        split,
    )
    prediction_path = args.output_root / "predictions" / filename
    rows = prediction_rows(
        predictions,
        data.records[split],
        config,
        spec,
    )
    write_csv(prediction_path, rows, PREDICTION_FIELDS)
    save_evaluation_metrics(
        args, config, run_metadata, evaluation_results[0], len(rows), filename
    )
    logger.info("Saved {} {} predictions to {}", len(rows), split, prediction_path)


def save_evaluation_metrics(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
    metrics: Mapping[str, float],
    sample_count: int,
    prediction_file: str,
) -> None:
    """Store one row of target, checkpoint, cohort, and split metric data."""
    split = args.split
    metric_values = {
        key.removeprefix(f"{split}/"): float(value)
        for key, value in metrics.items()
    }
    job_id = slurm_job_id()
    task_id = str(args.task_index)
    output_name = prediction_file.replace(
        f"_{split}_predictions.csv", f"_{split}_metrics.csv"
    )
    row = {
        "target": config["target_column"],
        "model": config["model_name"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_selection_split": run_metadata["selection_split"],
        "best_validation_epoch": run_metadata["best_validation_epoch"],
        f"{split}_sample_count": sample_count,
        "prediction_file": prediction_file,
        "slurm_job_id": job_id,
        "slurm_task_id": task_id,
        **metric_values,
    }
    metric_path = args.output_root / "metrics" / output_name
    write_csv(metric_path, [row])
    logger.info("Saved test metrics to {}", metric_path)


def main() -> None:
    """Run one checkpoint-only split evaluation and local exports."""
    args = parse_args()
    config, spec = load_run_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing CPU fallback")
    run_metadata = checkpoint_metadata(args.checkpoint)
    data = SuryaDataModule(config, spec)
    data.setup(args.split)
    logger.info("Prepared {} cohort: {}", args.split, data.cohort[args.split])
    evaluate_checkpoint(args, config, spec, data, run_metadata)


if __name__ == "__main__":
    main()
