# Training and benchmarking

Model-fitting entry points and a utility for measuring quantile-regression batch-size performance.

- `benchmark_qr_batch_size.py`: measures throughput and memory use across requested QR batch sizes and precision settings.
- `train_vit_classifier.py`: trains the configured ViT flare-classification model.
- `train_vit_quantile_regression.py`: trains the configured ViT quantile-regression model.

Use the YAML configurations under `configs/`; training commands are documented in the repository [README](../../README.md).
