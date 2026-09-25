# Severe-event binary metrics

## Purpose and definitions

This non-training evaluation adds binary M+/X+ metrics to completed saved forecasts without changing source predictions, checkpoints, targets, or frozen splits. The corrected broad mapping is `FQ=0, A=1, B=2, C=3, M=4, X=5`; M+ means M/X and X+ means X only.

Reported decision metrics are TP, FP, TN, FN, positive precision, positive recall/POD, positive F1, negative F1, binary macro-F1, balanced accuracy, specificity, FPR, FNR, TSS, and HSS. POD and positive-class recall are numerically identical here and are retained as alternate solar-forecast labels, not separate evidence. TSS is `TP/(TP+FN)-FP/(FP+TN)`; HSS is `2(TP*TN-FN*FP)/((TP+FN)(FN+TN)+(TP+FP)(FP+TN))`.

QR q50 AUROC/AUPRC are threshold-independent ranking metrics. TSS, HSS, F1, macro-F1, POD, and balanced accuracy are threshold-dependent: each threshold is selected only from validation q50 scores by maximum TSS, then positive F1, then POD, then higher threshold. It is frozen before test evaluation; TEST labels, metrics, and sweeps never select it.

Only `max_peak_flux` supports raw physical thresholds: M1 `1e-5 W/m2` and X1 `1e-4 W/m2` on q50_raw. Cumulative peak flux is an aggregate and RXFI has no single-event GOES cutoff.

## Inputs and artifact boundary

Project 1 used `outputs/project3_exports/{max_peak_flux,cumulative_peak_flux,max_rxfi,cumulative_rxfi}/{validation,test,leaky_validation}.csv`. Project 2 QR used the four saved `outputs/project2/vit_small_224_*_cumulative_peak_flux_qr/predictions/*_predictions.csv` sets.

Corrected classification used only the four `*_max_flare_class_ordinal6/test_confusion_matrix.csv` files. Each matrix row/column label was checked against its sibling `class_mapping.json` before M+/X+ collapse. No row predictions/probabilities/logits, validation matrix, or leaky-validation matrix were saved; therefore classification severe metrics are TEST-only and classification AUROC/AUPRC/non-test severe metrics are explicitly unavailable, never filled or approximated.

New outputs are in [`outputs/severe_event_binary_metrics/`](../../outputs/severe_event_binary_metrics/): machine-readable summary tables, `artifact_availability.csv`, `physical_threshold_evaluation.csv`, and exact validation threshold sweeps.

## TEST results

Every TEST evaluation has 27,980 rows: M+ has 8,596 positives/19,384 negatives and X+ has 968 positives/27,012 negatives.

### Project 1 QR (validation-selected threshold)

| Target | Event | AUPRC | AUROC | TSS | HSS | F1+ | POD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max peak flux | M+ | .655 | .854 | .580 | .493 | .686 | .904 |
| Max peak flux | X+ | .093 | .794 | .471 | .065 | .125 | .939 |
| Cumulative peak flux | M+ | .715 | .877 | .595 | .501 | .693 | .926 |
| Cumulative peak flux | X+ | .148 | .833 | .517 | .082 | .140 | .915 |
| Max RXFI | M+ | .613 | .828 | .523 | .450 | .657 | .849 |
| Max RXFI | X+ | .108 | .794 | .470 | .067 | .127 | .921 |
| Cumulative RXFI | M+ | .581 | .811 | .501 | .433 | .646 | .827 |
| Cumulative RXFI | X+ | .095 | .776 | .436 | .058 | .119 | .924 |

Selected transformed q50 thresholds (M+/X+) are max peak flux 2.697/2.733, cumulative peak flux 3.083/3.325, max RXFI .568/.578, and cumulative RXFI .986/.971. Cumulative peak flux leads the listed Project 1 metrics, while rare X+ results still have low HSS/F1 despite high POD.

### Project 2 corrected direct classification

| Input | Event | F1+ | Macro-F1 | POD | TSS | HSS | Balanced accuracy | AUROC/AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hmi_m | M+ | .034 | .427 | .018 | .013 | .018 | .507 | unavailable |
| hmi_m | X+ | .000 | .491 | .000 | .000 | .000 | .500 | unavailable |
| aia131 | M+ | .389 | .614 | .264 | .222 | .270 | .611 | unavailable |
| aia131 | X+ | .000 | .491 | .000 | .000 | .000 | .500 | unavailable |
| aia193 | M+ | .230 | .524 | .143 | .099 | .125 | .549 | unavailable |
| aia193 | X+ | .000 | .491 | .000 | .000 | .000 | .500 | unavailable |
| all13 | M+ | .364 | .595 | .253 | .192 | .231 | .596 | unavailable |
| all13 | X+ | .020 | .501 | .010 | .010 | .019 | .505 | unavailable |

All13 is the only direct classifier with nonzero X+ recall: 10/968. This result is limited to saved TEST matrices.

### Project 2 cumulative-peak-flux QR (validation-selected threshold)

| Input | Event | AUPRC | AUROC | TSS | HSS | F1+ | POD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hmi_m | M+ | .637 | .842 | .560 | .483 | .677 | .870 |
| hmi_m | X+ | .086 | .777 | .391 | .045 | .108 | .958 |
| aia131 | M+ | .664 | .853 | .538 | .439 | .661 | .929 |
| aia131 | X+ | .110 | .795 | .453 | .076 | .135 | .826 |
| aia193 | M+ | .606 | .808 | .461 | .359 | .622 | .940 |
| aia193 | X+ | .073 | .759 | .371 | .044 | .106 | .930 |
| all13 | M+ | .722 | .878 | .597 | .505 | .695 | .921 |
| all13 | X+ | .121 | .824 | .492 | .071 | .131 | .932 |

Selected q50 thresholds (M+/X+) are hmi_m 2.953/2.688, aia131 2.953/3.266, aia193 2.750/2.828, and all13 3.094/3.219. All13 leads the listed QR metrics, but high X+ POD is accompanied by low F1/HSS and must be interpreted with the 968-event support.

## Physical max-peak-flux result and limitations

At physical M1, Project 1 max-peak-flux TEST has TSS .465, HSS .468, F1+ .629, and POD .622. At physical X1 it predicts no TEST positives (POD/F1/TSS/HSS = 0), despite q50 ranking AUROC .794 and AUPRC .093. Physical and validation-selected thresholds answer different questions.

This is one seed and a 24-hour horizon. TSS combines TPR and FPR but is not completely immune to rare-event consequences; support and confusion counts remain necessary context. Threshold-dependent QR results are validation-selected and frozen before TEST evaluation; test labels are never used to optimize them.
