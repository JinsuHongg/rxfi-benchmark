# Regression Target and Operational Skill Experiment Design

## Purpose and research questions

This plan evaluates regression targets as a main research contribution. It
keeps two claims separate:

- **Learnability:** how accurately can a model predict each continuous target
  from the available solar image inputs?
- **Operational skill:** how well can scores from each target identify
  conventional M+ and X+ forecast windows?

A target may be easier to learn without improving M+/X+ skill. The results
will be reported as separate analyses and will not be collapsed into one
overall score.

The work has two phases. The target/background distribution audit is
exploratory. Background data from the test period have been inspected, but no
model results for the proposed regression targets have been examined. Freeze
the target conditions, model-selection rules, and operational thresholds
before evaluating model predictions on test. Use test once for the final
model comparison and do not use its outcomes to revise the protocol.

## Target conditions

The forecast window is the half-open interval `[s, s + 24 h)`. Eligible XRS-B
samples are finite, positive, science-quality one-minute measurements
(`xrsb_flag == 0`). No flare-catalog masking is applied to these XRS-only
targets.

For background window length `W` and lower-tail percentage `q` (10 or 50),
let `R` be the eligible observations in `[s - W, s)`. Sort them in ascending
order and let `k = ceil((q / 100) * len(R))`. The background `B_(W,q)(s)` is
the arithmetic mean of the first `k` values. The four background conditions
are:

| Condition | Background interval | Lower-tail samples averaged |
| --- | --- | ---: |
| `BG-2h-10` | `[s - 2 h, s)` | 10% |
| `BG-2h-50` | `[s - 2 h, s)` | 50% |
| `BG-24h-10` | `[s - 24 h, s)` | 10% |
| `BG-24h-50` | `[s - 24 h, s)` | 50% |

The ten target conditions are the two background-independent targets plus
each of the two deviation targets under all four backgrounds:

| Target family | Conditions | Definition |
| --- | ---: | --- |
| `max_xflux` | 1 | `max_i F_i` |
| `fluence` | 1 | `sum_i F_i` |
| `max_background_deviation` | 4 | `max_i (F_i - B_(W,q)(s))` |
| `total_background_deviation` | 4 | `sum_i (F_i - B_(W,q)(s))` |

The split outputs use concise file names in `data/derived/`:
`train.csv`, `validation.csv`, `leaky_validation.csv`, and `test.csv`. Target
columns preserve the names in the table, with deviation variants suffixed by
their background ID (for example, `max_background_deviation_2h_q10`). The
reproducible generator and its command are documented in
[`scripts/data/README.md`](../../scripts/data/README.md).

The deviation is signed. Negative sample deviations and negative window
totals are retained. Missing XRS samples are not interpolated.

### Sum scaling convention

For regression, `fluence` and `total_background_deviation` are stored as
unscaled one-minute sample sums; the factor of 60 seconds is omitted. It
changes the physical unit conversion, not the information in the target. If
results need physical fluence units, multiply the sum by 60 seconds to express
it in J m^-2. The unscaled stored value is a cadence sum and should not be
described as an SI fluence. Any target normalization or inverse transform must
be fitted on the training split and applied consistently to every split.

## Experiment A: background and target distributions

**Type:** exploratory.

**Objective:** quantify how background window length and lower-tail percentage
change background estimates, target distributions, data availability, and
their association with conventional flare classes.

For every split and each background condition, report background availability,
coverage, median, interquartile range, selected upper quantiles, and the
distribution of the two signed deviation targets. Report negative, zero,
positive, and missing target counts. Stratify descriptive summaries by
NOAA-derived M+/X+ window status, while making clear that this analysis does
not establish model learnability or operational skill.

Use identical valid XRS sample rules across the four background conditions.
Record per-window coverage in both background and forecast intervals. Freeze
any minimum-coverage exclusion threshold before model training; until that
threshold is specified, show coverage-stratified sensitivity summaries
instead of silently dropping low-coverage windows.

## Experiment B: regression learnability

**Type:** confirmatory after the protocol and final cohort are frozen.

**Claim tested:** one or more proposed regression targets are more learnable
from the selected image inputs than the conventional maximum X-ray target.

**Hypothesis:** at least one of the proposed targets has higher held-out
baseline-relative regression skill than `max_xflux` on the common valid
sample cohort, under the same model and training budget.

Use the intersection of rows with valid labels for all ten target conditions
as the primary cohort. This keeps target comparisons on the same examples.
Also report each target's full available cohort as a secondary analysis,
with sample counts and coverage so that availability differences remain
visible.

Keep image inputs, model architecture, optimizer search budget, training
budget, stopping rule, and random seeds fixed across targets. Fit each target's
transform using train only. Compare errors in that prespecified transformed
space and report skill relative to a train-only constant-median baseline;
include normalized MAE (divide by the train target IQR), R-squared, and
Spearman correlation. Report raw-unit MAE/RMSE for interpretation within a
target, not as the cross-target ranking criterion. Use multiple fixed seeds
and report the seed-level distribution rather than selecting a favorable run.

Use `validation` for checkpoint and model-selection decisions. Keep
`leaky_validation` as a separately reported leakage-sensitivity diagnostic;
do not use it as the primary threshold or calibration set. Test outcomes have
not been examined for the proposed target models. Freeze all target and model
choices before evaluating test predictions.

**Primary metric:** baseline-relative skill in the prespecified transformed
target space on the common cohort.

**Guardrails:** normalized MAE, R-squared, Spearman correlation, valid-row
coverage, and performance across coverage strata. Use paired time-block
bootstrap intervals for target-versus-`max_xflux` differences to account for
serial dependence among hourly samples and overlapping forecast windows.
Correct for the planned multiple comparisons across candidate targets.

**Falsification:** the learnability claim is not supported if candidate
targets do not improve the prespecified baseline-relative skill over
`max_xflux`, or if any apparent improvement is not robust across seeds and
time-block uncertainty intervals.

## Experiment C: M+/X+ operational skill

**Type:** confirmatory on the frozen test split, provided target definitions,
model settings, and threshold rules are fixed before test predictions are
examined.

**Objective:** compare the operational value of regression scores from all
target conditions for M+ and X+ forecast-window detection.

Define an M+ positive window as one containing at least one NOAA flare-report
event with `start_time` in `[s, s + 24 h)` and peak XRS-B irradiance at or above
`1e-5 W m^-2`. Define X+ using `1e-4 W m^-2`. Use the same catalog release,
event membership rule, and positive-window labels for every target condition.
The catalog is reserved for these operational outcomes; it is not used to
construct the ten XRS-only regression targets.

Use the continuous regression score as the ranking score. Do not require a
probability calibration mapping between target scales. Select one decision
threshold per target and event definition on `validation`, using the same
threshold-selection rule for every target, then freeze it before evaluation.
The validation set may also select checkpoints; this reuse makes validation
metrics selection-biased, so report them as development metrics. The
thresholded final metric must be computed on an untouched evaluation split.

### Metrics

The primary operational metrics are:

- **TSS:** `TP / (TP + FN) - FP / (FP + TN)`.
- **HSS:** `2 * (TP * TN - FN * FP) / ((TP + FN)(FN + TN) + (TP + FP)(FP + TN))`.
- **CSS:** the geometric mean `sqrt(TSS * HSS)` when both TSS and HSS are
  non-negative; otherwise report CSS as undefined and retain TSS/HSS.

Select the validation threshold by maximum CSS among thresholds where CSS is
defined; break ties by higher TSS, then higher HSS, then the higher threshold.
Report this selection rule and the selected threshold for each target and
event definition. Also report the threshold-independent PR-AUC and ROC-AUC,
and the following secondary threshold-dependent metrics: positive F1,
macro-F1, recall/POD, false alarm rate, and precision. Include positive and
negative supports and the full confusion matrix so rare X+ outcomes are
interpretable.

For `max_xflux`, additionally report the direct physical thresholds
`1e-5 W m^-2` and `1e-4 W m^-2` on inverse-transformed predictions. These are
an interpretable XRS threshold-exceedance reference. It is a distinct outcome
from the primary catalog event label, because a high sample can belong to a
flare that started before the window. The validation-selected threshold
comparison against the common catalog event labels remains the primary
cross-target protocol.

## Split roles and leakage controls

| Split | Role in this design |
| --- | --- |
| Train | Fit model parameters and train-only target transforms |
| Validation | Select checkpoint/model settings and per-target M+/X+ decision thresholds |
| Leaky validation | Leakage-sensitivity diagnostic only; not threshold selection or independent validation |
| Test (2020–2024) | Final evaluation of frozen models and thresholds |

The threshold is selected separately for M+ and X+ from validation scores,
then applied unchanged to test. Never select a threshold, target definition,
transform, cohort rule, or model from test outcomes. The leaky-validation
split is not a substitute for a clean calibration set because its temporal
placement can expose shared events or nearby windows; use it to quantify
sensitivity to that condition.

Overlapping hourly windows can contain the same flare. This plan's primary
operational estimand is therefore window-level M+/X+ skill. Event-level
deduplication, alert-episode merging, and region-level scoring are deferred
until a separate event-matching protocol is specified; do not mix those
results into the primary window-level comparison.

## Data provenance

The four XRS-only target definitions and generation provenance are documented
in [xrs_only_target_definitions.md](xrs_only_target_definitions.md). NOAA
catalog fields and event timing are documented in
[noaa_xrs_catalog_and_rxfi_definition.md](noaa_xrs_catalog_and_rxfi_definition.md).

## Unresolved items before execution

- Freeze the minimum background and forecast-window coverage rule.
- Freeze target definitions, model settings, and validation thresholds before
  opening test model predictions.
- Confirm that image inputs and NOAA event outcomes cover the full test period.
- Freeze the image model/configuration and the number of training seeds.
- Validate CSS implementation and undefined cases against the project's
  metric code before using it for threshold selection.
- Specify event-level alert matching only if event-level results are later
  needed; it is not part of this experiment's primary analysis.
