# Project 3 prediction exports

Tools and contract for creating and validating prediction files for Project 3.

- `export_project3_predictions.py`: exports selected model predictions with source-row identity and metadata.
- `project3_export_contract.py`: standalone schema and validation helpers shared with downstream consumers.
- `project3_export_loader_example.py`: example code for loading and consuming an export.
- `validate_project3_exports.py`: validates generated export files against the Project 3 contract.

From the repository root, run `python scripts/exports/project3/export_project3_predictions.py --all` and `python scripts/exports/project3/validate_project3_exports.py --all`.
