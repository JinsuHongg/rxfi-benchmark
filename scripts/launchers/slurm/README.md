# Project 1 Slurm launchers

- `compute_input_stats.sbatch`: computes signed-log input channel statistics from the training rows in `surya_index/train_targets.csv` and saves them to `outputs/input_normalization_stats/train_channel_stats.json`. Submit it from the repository root with `sbatch scripts/launchers/slurm/compute_input_stats.sbatch`.
- `run_qr_array.sbatch`: four-index Slurm array template for Project 1 quantile-regression runs. Set cluster-specific partition, account, time, and memory options before submission.
