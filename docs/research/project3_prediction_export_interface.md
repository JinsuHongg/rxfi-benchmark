# Project 3 Prediction Export Interface

`rxfi-benchmark` owns targets, inputs, predictive models, QR training, and prediction exports. `ordinal-cqr` owns OCQR, ordinal prediction sets, and coverage/efficiency evaluation. This interface exports predictions and labels only; it does not implement a conformal method.

Each directory in `outputs/project3_exports/` has `validation.csv`, `leaky_validation.csv`, `test.csv`, and `metadata.json`, using `project3_prediction_export_v1`. Rows contain `dataset`, `experiment_id`, `split`, `original_row_index`, `timestamp`, `target_name`, raw/transformed target and q05/q50/q95 values, preserved `max_flare_class`/`max_flare_class_raw`, and `ordinal_class`/`ordinal_class_index`.

The row key is `(split, original_row_index, timestamp)`, never CSV order. Original row index is zero-based within each physical split. Labels use `FQ < A < B < C < M < X` mapped to `0..5`; A is retained despite no current samples and malformed labels are rejected. `leaky_validation` is physically preserved and is only a preliminary metadata alias for calibration; `validation` remains checkpoint selection and `test` conformal evaluation.

Transforms: `max_peak_flux=log10(raw/1e-8)`, `cumulative_peak_flux=log10(1+raw/1e-8)`, and both RXFI targets use `log10(1+raw)`. Metadata carries transforms, horizon, model/image details, seed, checkpoint selection, hashes, class mapping, roles, and source commit. Consumers must reject unknown versions.

```text
rxfi-benchmark QR predictions → standardized export → ordinal-cqr loader
→ conformal calibration → ordinal prediction sets → coverage / efficiency
```

Use `python scripts/export_project3_predictions.py --all` and `python scripts/validate_project3_exports.py --all`. Begin proposal work with `max_peak_flux` and `cumulative_peak_flux`; the RXFI targets remain secondary and available.
