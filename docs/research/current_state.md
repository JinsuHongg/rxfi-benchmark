# Current Research State

## Project 1: NOAA XRS catalog and RXFI

- NOAA/NCEI science-quality Level-2 GOES Flare Report verification: completed; it is the canonical candidate catalog for flare peak and event-specific background XRS-B irradiance, pending release/checksum pinning for the thesis dataset.
- RXFI definition freeze: completed as `RXFI = (xrsb_irrad - background_irrad) / background_irrad` for valid positive inputs.
- RXFI distribution and imbalance analysis: completed as an exploratory, read-only analysis of the frozen supplied splits. It validates the split calendar rules and documents class imbalance, RXFI tails, split shift, and the confirmed 24-hour dataset-contract event association.
- Dataset contract: confirmed. Each frozen split timestamp uses the half-open future window `[t, t + 24 hours)` and its maximum NOAA XRS-B peak event for the maximum-peak association. The split files do not store row-level NOAA IDs, which is a traceability limitation but not an unresolved matching assumption.
- Four-target 24-hour generation: completed. Separate derived files preserve every frozen split row, timestamp, and order; they provide maximum/cumulative peak-flux and RXFI targets plus event provenance under the confirmed contract.
- Validation note: generated NOAA science-quality `max_flare_class` differs from the supplied historical class string for many pre-GOES-R rows. This is recorded as legacy-label compatibility evidence and does not alter frozen split labels or membership.
- Historical GOES label mismatch audit: completed. Broad ordinal-class agreement is 79.74–83.45% in historical validation-style splits and 95.12% in test; high continuous-flux correlation supports a historical source-version compatibility difference, with a documented no-flare/flare inclusion subset and no available legacy event IDs for exact-event proof.
- Next task: run the minimal ViT-Small 224x224 downstream experiment using the frozen split timestamps and generated NOAA targets, while documenting the legacy-label compatibility caveat.
