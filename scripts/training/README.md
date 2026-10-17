# Training and benchmarking

Model-fitting entry points and a utility for measuring quantile-regression batch-size performance.

- `compute_input_stats.py`: computes per-channel population mean/std after signed `log1p`, using only rows from the training index CSV. Its default output is `outputs/input_normalization_stats/train_channel_stats.json`.
- `compute_target_stats.py`: computes train-only, per-target scaled-asinh and population z-score statistics from the common cohort of rows with all ten finite targets and matching timestamps in the configured Zarr store.
- `benchmark_qr_batch_size.py`: measures throughput and memory use across requested QR batch sizes and precision settings.
- `train_vit_classifier.py`: trains the configured ViT flare-classification model.
- `train_vit_quantile_regression.py`: trains the configured ViT quantile-regression model.

Run the stats script from the repository root before training:

```bash
python scripts/training/compute_input_stats.py \
  --zarr-path /scratch/users/jhong36/surya-bench-224-v2.zarr \
  --index-csv /scratch/users/jhong36/data/surya_index/train_targets.csv \
  --output outputs/input_normalization_stats/train_channel_stats.json
```

The classifier and quantile-regression dataloaders apply `sign(x) * ln(1 + abs(x))`, then normalize each selected channel with these training-split statistics. Configurations use `normalization_stats_path` to select the JSON file. The statistics file records the transform, channel order, training row count, source paths, and input checksum. Use statistics computed from the same training data and channel order as the run.

After generating the ten XRS-only target split CSVs, compute target statistics
on the interarea Zarr with the Slurm launcher:

```bash
mkdir -p logs/target_stats
sbatch scripts/launchers/slurm/compute_target_stats.sbatch
```

The launcher reads `data/derived/train.csv`, matches its timestamps to the
Zarr time arrays, and writes
`outputs/target_normalization_stats/train_target_stats.json`. Target values
use `asinh(y / s)`, where `s` is the training-cohort median absolute value;
the artifact also records the transformed mean and population standard
deviation for per-target z-score normalization.

The XRS-only QR configurations under `configs/xrs_only/` train one model for
each of the ten target columns. They read the generated split CSVs directly,
use the interarea Zarr and input stats, and apply each target's train-fitted
asinh and z-score values. Submit the ten-run array after generating the CSVs
and target stats:

```bash
mkdir -p logs/xrs_qr
sbatch scripts/launchers/slurm/run_xrs_only_qr_array.sbatch
```

Use the YAML configurations under `configs/`; training commands are documented in the repository [README](../../README.md).
