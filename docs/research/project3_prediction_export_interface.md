# Project 3 Prediction Export Interface

`rxfi-benchmark` owns targets, inputs, predictive models, QR training, and prediction exports. `ordinal-cqr` owns OCQR, ordinal prediction sets, and coverage/efficiency evaluation. This interface exports predictions and labels only; it does not implement a conformal method.

The current split-specific RXFI target data use the selected lower-50% background. Existing Project 3 prediction exports still contain historical NOAA `background_irrad` labels; if regenerated, their metadata and interface version must identify the lower-50% background method so consumers do not mix label versions. The current event-level calculation is documented in [lower50_background_rxfi.md](lower50_background_rxfi.md).

New exports are written under `outputs/project3_exports/`; each target directory has `validation.csv`, `leaky_validation.csv`, `test.csv`, and `metadata.json`. Existing six-class exports from the affected 224×224 runs are retained under `outputs/legacy_224x224/project3_exports/` as `project3_prediction_export_v1` and are quarantined for provenance. Newly generated exports use `project3_prediction_export_v2`. Rows contain `dataset`, `experiment_id`, `split`, `original_row_index`, `timestamp`, `target_name`, raw/transformed target and q05/q50/q95 values, preserved `max_flare_class`/`max_flare_class_raw`, and `ordinal_class`/`ordinal_class_index`.

The row key is `(split, original_row_index, timestamp)`, never CSV order. Original row index is zero-based within each physical split. New v2 labels merge `FQ` and `A*` as `FQ/A=0`, followed by `B=1`, `C=2`, `M=3`, and `X=4`; malformed labels are rejected. The validator continues to read existing v1 exports with their historical `FQ=0, A=1, B=2, C=3, M=4, X=5` mapping. `leaky_validation` is physically preserved and is only a preliminary metadata alias for calibration; `validation` remains checkpoint selection and `test` conformal evaluation.

Transforms: `max_peak_flux=log10(raw/1e-8)`, `cumulative_peak_flux=log10(1+raw/1e-8)`, and both RXFI targets use `log10(1+raw)`. Metadata carries transforms, horizon, model/image details, seed, checkpoint selection, hashes, class mapping, roles, and source commit. Consumers must reject unknown versions.

```text
rxfi-benchmark QR predictions → standardized export → ordinal-cqr loader
→ conformal calibration → ordinal prediction sets → coverage / efficiency
```

Use `python scripts/exports/project3/export_project3_predictions.py --all` and `python scripts/exports/project3/validate_project3_exports.py --all`. Begin proposal work with `max_peak_flux` and `cumulative_peak_flux`; the RXFI targets remain secondary and available.
