---
name: rxfi-experiment-ops
description: Run and monitor approved RXFI benchmark training experiments, including Slurm sweeps and validation-based selection.
---

# RXFI experiment operations

Use this skill for executing or checking training experiments in rxfi-benchmark.
Use the experiment-designer skill when the research design itself is still open.

## Before execution

- Work from the experiment plan the user approved. Confirm target, model,
  data split, input channels, seeds, hyperparameter grid, and output location.
- Use the `rxfi-benchmark` Conda environment and the repository's existing
  training and evaluation entrypoints.
- Check that checkpoint selection and LR/WD selection use the same primary
  validation metric. The project default is full `validation/pinball_loss`,
  averaged over q05, q50, and q95. For a sweep, select LR/WD by the mean
  best-checkpoint validation pinball loss across the planned seeds.
- Do not use test metrics to select checkpoints, hyperparameters, or channels.
  If test results have already influenced the experiment, label later test
  comparisons exploratory.
- Do not use leaky validation unless the user explicitly includes it.

## Submit and monitor

- Submit Slurm jobs only after the user directs execution.
- Keep concurrent experiment tasks at or below three unless the user changes
  that limit.
- Check job state and logs with concise commands. Report pending, running,
  completed, and failed tasks separately; do not infer completion from
  missing log output.
- Keep training runs in W&B project `rxfi-regression` and test runs in
  `rxfi-regression-test`, when W&B is enabled in the approved plan.

## Preserve and report results

- Preserve the selected checkpoint and per-seed validation metrics.
- Save sample-level predictions for requested final validation or test
  evaluations, with filenames that identify target, model, and seed.
- Summarize the selected settings and metrics across all planned seeds,
  including mean and sample standard deviation. Report failed or missing
  runs and cohort counts.
- Do not modify source files, install packages, push commits, or submit
  additional jobs beyond the approved scope.
