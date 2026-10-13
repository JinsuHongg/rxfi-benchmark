# Paper background calculation package

Python package used by the background calculation command-line script.

- `__init__.py`: exposes the package module namespace.
- `calculation.py`: indexes flare intervals and computes capped and uncapped 24-hour event backgrounds.
- `noaa.py`: downloads, caches, verifies, and reads checksum-pinned NOAA/NCEI inputs.
- `runner.py`: orchestrates input validation, calculation, provenance, and output files.
