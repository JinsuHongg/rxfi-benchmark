# Experiment launchers

Shell and scheduler entry points for training jobs. Invoke them from the repository root unless a job script documents otherwise.

- `run_all_qr_experiments.sh`: launches the four Project 1 QR experiments with explicit GPU assignments.
- `slurm_vit_small_224_max_flare_class.sbatch`: submits a single ViT flare-classification job to Slurm.
- [`slurm/`](slurm/README.md): Project 1 QR Slurm array template.
- [`project2/`](project2/README.md): Project 2 local and Slurm launchers.
