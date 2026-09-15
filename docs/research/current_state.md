# Current Research State

## Project 1: NOAA XRS catalog and RXFI

- NOAA/NCEI science-quality Level-2 GOES Flare Report verification: completed; it is the canonical candidate catalog for flare peak and event-specific background XRS-B irradiance, pending release/checksum pinning for the thesis dataset.
- RXFI definition freeze: completed as `RXFI = (xrsb_irrad - background_irrad) / background_irrad` for valid positive inputs.
- RXFI distribution and imbalance analysis: completed as an exploratory, read-only analysis of the frozen supplied splits. It validates the split calendar rules and documents class imbalance, RXFI tails, split shift, and candidate event matching.
- Important unresolved issue: the split CSVs do not include NOAA event IDs. The analysis uses a clearly labeled 24-hour maximum-peak candidate association; this must be validated against the dataset-generation contract before RXFI becomes a downstream per-sample target.
- Next task: prepare the minimal downstream ML experiment using the frozen splits and RXFI analysis, only after resolving or formally accepting the sample-to-NOAA event matching contract.
