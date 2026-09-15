# Current Research State

## Project 1: NOAA XRS catalog and RXFI

- Verified NOAA/NCEI's science-quality Level-2 GOES Flare Report as the canonical candidate catalog for flare peak and event-specific background XRS-B irradiance.
- Accepted the catalog as the canonical Project-1 source, subject to pinning a file release and checksum for the thesis analysis.
- Frozen the initial eligible-event definition as `RXFI = (xrsb_irrad - background_irrad) / background_irrad`.
- Remaining questions: confirm historical `background_irrad` provenance with NOAA; specify final temporal cutoff and pre-specified source/saturation/sequential-flare sensitivity analyses.
- Next task: pin the target catalog release and build a non-ML event-level data-quality audit using the frozen fields and rules.
