# ViT-Small Quantile Regression for Maximum XRS-B Flux

## Purpose

This pipeline prepares the continuous predictive model required before Ordered Conformalized Quantile Regression (OCQR). It is independent of the concurrently trained flare-classification checkpoint and can be smoke-tested from random ViT-Small initialization.

## Input and target

SuryaBench contributes only 13-channel, 224x224 SDO images, timestamps, and membership in the immutable train, validation, test, and leaky-validation splits. The supervised target is exclusively regenerated NOAA/NCEI `max_peak_flux`, not any legacy SuryaBench label.

The regression response is

\[
z = \log_{10}(\texttt{max\_peak\_flux}),
\]

where flux is in W/m2 and no epsilon is added. For flare windows, the physical value is the maximum NOAA flare-summary XRS-B peak in `[t, t + 24h)`. For `FQ` windows, it is the maximum valid science-quality NOAA XRS-B one-minute flux in the same window. This retains the confirmed distinction between an event-catalog FQ class and a continuous irradiance maximum.

## Missing-target policy

The 105 FQ rows with no valid one-minute maximum are retained in the frozen source and derived CSVs, but are explicitly excluded from regression datasets by a finite-and-positive target mask. They are never assigned zero or imputed. The pipeline writes the valid and excluded count per split into `dataset_integrity.json` before model construction.

Image availability is audited separately from target validity. On the current local SuryaBench Zarr store, 49,104 valid-target timestamps have no matching image record (the same broad availability issue recorded by the classification baseline), leaving 44,962 train, 2,422 validation, 27,980 test, and 3,755 leaky-validation image-target pairs. These rows are retained in every frozen CSV and reported in `missing_or_unreadable_images.csv`; their exclusion is never interpreted as a target value or a split rewrite.

## Model and optimization

The configured model is `vit_small_patch16_224` with 13 input channels, 224-pixel input, random initialization (`pretrained: false`), and three raw outputs in fixed order: `q05`, `q50`, `q95`. It uses AdamW (learning rate `1e-4`, weight decay `0.01`), batch size 4, one seed (0), and ten configured epochs. FP16 was attempted on the GTX 1660 Ti but generated a non-finite gradient during smoke testing, so the checked configuration uses the stable FP32 fallback. The checkpoint metric is minimum validation pinball loss; test data are never used for selection.

For each quantile \(\tau \in \{0.05, 0.50, 0.95\}\), the loss is

\[
\max\{\tau(y-q_\tau), (\tau-1)(y-q_\tau)\},
\]

averaged over samples and quantiles. Raw output order is retained: no sorting or monotonicity constraint is applied. Validation reports pairwise quantile-crossing rates, per-quantile pinball losses, q50 MAE/RMSE, and empirical fractions `target <= q05/q50/q95`; these fractions are diagnostics, not conformal coverage claims.

## OCQR interface and calibration status

There is no OCQR implementation or fixed calibration partition in the current repository, so no adapter or calibration procedure is created here. Full training exports raw prediction CSVs under `outputs/vit_small_224_max_peak_flux_qr/predictions/` with the following schema:

```text
split,original_row_index,timestamp,max_peak_flux,z,max_flare_class,q05,q50,q95
```

`prediction_schema.json` separately freezes quantile order `[0.05, 0.50, 0.95]`. These values are sufficient for a later OCQR adapter, which must define its nonconformity score and use an explicitly fixed calibration partition without splitting validation or test rows ad hoc. Calibration partitioning is therefore **unresolved** rather than invented.

## Reproducibility and outputs

Run a checkpoint-free smoke test:

```bash
conda run -n rxfi-benchmark python scripts/train_vit_quantile_regression.py --smoke-only
```

The smoke test performs 32 training iterations, forward/backward/optimizer steps, finite-loss and finite-gradient checks, crossing diagnostics, and peak CUDA-memory measurement. A full run uses the same script without `--smoke-only`; it writes the resolved configuration, integrity metadata, training log, validation metrics, checkpoint metadata, prediction exports, and learning curve. Checkpoint files remain gitignored.

## Limitations and next step

This is one seed, one ViT size, one 224x224 resolution, no pretrained 13-channel initialization, and no hyperparameter search. It does not conduct OCQR calibration or evaluation. The next step is: **Train the quantile-regression model, export predictions, and run the existing OCQR calibration/evaluation pipeline** once a fixed calibration partition and OCQR implementation are available.
