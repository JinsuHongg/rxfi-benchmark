# Quarantined 224×224 experiment outputs

These outputs were generated from model inputs resized to 224×224. A defect was
identified in that input data, so the results are retained for provenance and
auditability but must not be reused as valid experiment results or cited as
current evidence. They have not been deleted.

Archived artifacts:

- `four_qr_target_analysis/`: cross-target model analysis and figures.
- `project2/`: preliminary multichannel model runs and summaries.
- `project2_channel_audit.json`: channel and cohort audit from those runs.
- `project3_exports/`: prediction exports from the affected runs.
- `severe_event_binary_metrics/`: metrics computed from affected saved model
  predictions.

Resolution-independent products, such as background and target-generation
audits, remain in their existing `outputs/` directories. New model runs and
exports should be generated into the normal output paths after the data pipeline
has been corrected. Keep this archive intact unless a replacement archive policy
is established.
