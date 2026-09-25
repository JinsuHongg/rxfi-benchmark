# Current Research State

## Project 1: NOAA XRS catalog and RXFI

- NOAA/NCEI science-quality Level-2 GOES Flare Report verification: completed; it is the canonical candidate catalog for flare peak and event-specific background XRS-B irradiance, pending release/checksum pinning for the thesis dataset.
- RXFI definition freeze: completed as `RXFI = (xrsb_irrad - background_irrad) / background_irrad` for valid positive inputs.
- RXFI distribution and imbalance analysis: completed as an exploratory, read-only analysis of the frozen supplied splits. It validates the split calendar rules and documents class imbalance, RXFI tails, split shift, and the confirmed 24-hour dataset-contract event association.
- Dataset contract: confirmed. Each frozen split timestamp uses the half-open future window `[t, t + 24 hours)` and its maximum NOAA XRS-B peak event for the maximum-peak association. The split files do not store row-level NOAA IDs, which is a traceability limitation but not an unresolved matching assumption.
- Four-target 24-hour generation: completed. Separate derived files preserve every frozen split row, timestamp, and order; they provide maximum/cumulative peak-flux and RXFI targets plus event provenance under the confirmed contract.
- No-flare continuous `max_peak_flux` refinement: completed and QC-validated. Event-free windows now use the positive maximum valid NOAA XRS-B 1-minute irradiance in the same `[t, t + 24 hours)` window; flare classes and all event-based cumulative/RXFI targets are unchanged. Windows without any valid 1-minute observation retain a missing maximum rather than an artificial zero.
- FQ XRS threshold-exceedance QC: completed. FQ remains an event-catalog label, not a hard instantaneous-flux bound. Among 27,559 valid FQ maxima, C1-or-higher values are uncommon (257; 0.93%), M1-or-higher values are very rare (13; 0.047%), and no X1-or-higher value occurs. Timing audits support nearby prior/outside-window flare semantics for many high values; unresolved cases are documented without relabeling. The conclusion is **PROCEED WITH DOCUMENTED CAVEAT** and supports `log10(max_peak_flux)` as the OCQR continuous target.
- Four-target ViT-Small quantile-regression configuration and cluster launch scripts: completed. The shared pipeline supports frozen config-driven transforms, q05/q50/q95 prediction exports, validation pinball-loss checkpoint selection, GPU/precision/batch/worker overrides, and a Slurm array template. It is calibration-ready, but the repository contains neither an OCQR implementation nor a fixed calibration partition.
- Validation note: generated NOAA science-quality `max_flare_class` differs from the supplied historical class string for many pre-GOES-R rows. This is recorded as legacy-label compatibility evidence and does not alter frozen split labels or membership.
- Historical GOES label mismatch audit: completed. Broad ordinal-class agreement is 79.74–83.45% in historical validation-style splits and 95.12% in test; high continuous-flux correlation supports a historical source-version compatibility difference, with a documented no-flare/flare inclusion subset and no available legacy event IDs for exact-event proof.
- QR-1 through QR-4 full training: completed.
- Four-target learnability analysis: completed.
- Severe-event relevance analysis: completed.
- Project 2 first-wave multichannel experiments: completed. Four cumulative-peak-flux QR runs, four original exact-string classification runs (audit only), and four corrected ordinal6 classification runs completed on the frozen common cohorts.
- Project 2 integrated preliminary analysis: completed. All13 is strongest across listed QR metrics; corrected classification is mixed, with AIA131 leading observed-class macro-F1/balanced accuracy and all13 retaining limited nonzero X recall.
- Severe-event binary evaluation: completed from saved outputs only. M+/X+ confusion-count metrics, validation-frozen QR thresholds, exact threshold curves, and scientifically limited max-peak-flux physical thresholds are under `outputs/severe_event_binary_metrics/`; corrected classification is TEST-only because only saved TEST matrices support exact collapse, and its AUROC/AUPRC remain unavailable without saved probabilities.
- Project 2 next task: review the preliminary findings and define the second wave (additional seeds first, then shorter horizons and selected channel groups).
- Project 1 next task: interpret the target comparison, freeze the conformal calibration partition, then run OCQR consistently across selected or all targets.
- Project 1: complete enough for proposal; standardized QR exports now provide the Project 3 boundary.
- Project 2: preliminary multichannel analysis complete.
- Project 3: repository boundary defined and prediction-export interface frozen; OCQR calibration/evaluation will move to `ordinal-cqr`.
- Next: validate Project 3 exports, consume them in `ordinal-cqr`, implement/reuse OCQR calibration there, and produce preliminary coverage/set-size results.
