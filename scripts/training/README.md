# Training and benchmarking

Model-fitting entry points and a utility for measuring quantile-regression batch-size performance.

- `compute_input_stats.py`: computes per-channel population mean/std after signed `log1p`, using only rows from the training index CSV. Its default output is `outputs/input_normalization_stats/train_channel_stats.json`.
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

Use the YAML configurations under `configs/`; training commands are documented in the repository [README](../../README.md).
