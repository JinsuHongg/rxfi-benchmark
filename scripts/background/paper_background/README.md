# Paper background calculation package

Python package used by the background calculation command-line script.

- `__init__.py`: exposes the package module namespace.
- `calculation.py`: indexes flare intervals and computes full-mean, start-capped, and optionally lower-tail 24-hour event backgrounds.
- `noaa.py`: downloads, caches, verifies, and reads checksum-pinned NOAA/NCEI inputs.
- `runner.py`: orchestrates input validation, calculation, provenance, and output files.

The selected lower-50% configuration is `configs/background_lower50.json`; use the command and output documentation in [`docs/research/lower50_background_rxfi.md`](../../../docs/research/lower50_background_rxfi.md).
