# Analysis and audits

Scripts for benchmark diagnostics, label audits, and model evaluation.

- `analyze_four_qr_targets.py`: summarizes and compares the four Project 1 quantile-regression targets.
- `analyze_rxfi_distribution.py`: analyzes RXFI distribution and target imbalance.
- `plot_four_target_distributions.py`: plots train-only histograms for max X-ray flux, fluence, and the two 24-hour q50 background-subtracted flux targets. Run with `conda run -n rxfi-benchmark python scripts/analysis/plot_four_target_distributions.py`; input paths and bin count can be overridden with CLI flags.
- `plot_v1_target_distributions.py`: plots train-only distributions for all ten V1 XRS regression targets.
- `compare_old_rxfi_and_background_deviation.py`: compares the supplementary
  legacy `max_old_rxfi` label with XRS-only `max_background_deviation`, including
  paired-window coverage, reference-sign cross-tabs, block-bootstrap rank
  association, and background-deviation and old-RXFI histograms stacked by the
  split's stored GOES class. See
  [`docs/research/old_rxfi_vs_background_deviation.md`](../../docs/research/old_rxfi_vs_background_deviation.md)
  for the protocol and interpretation.
- `audit_fq_xrs_threshold_exceedance.py`: audits XRS threshold exceedances in FQ/no-flare samples.
- `audit_historical_goes_label_mismatch.py`: compares historical GOES labels against the current catalog.
- `evaluate_severe_event_binary_metrics.py`: evaluates severe-event ranking and binary metrics.
- `severe_event_metrics.py`: shared metric and validation-threshold functions used by evaluation code.
- [`project2/`](project2/README.md): Project 2 channel and preliminary-result analyses.
