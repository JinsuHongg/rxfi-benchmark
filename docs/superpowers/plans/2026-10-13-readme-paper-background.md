# README Background Method and Reproduction Instructions Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the project README to explain and cite the selected paper-based background method and show how to run its core reproducibility script.

**Architecture:** Update only the root `README.md`, keeping the detailed methodology and calculation results in their existing research documents. The README will clearly distinguish the paper-based 24-hour background experiment from the NOAA `background_irrad` field still used by existing generated targets.

**Tech Stack:** Markdown documentation, existing Python CLI, pinned JSON config and input lock.

**Spec:** User request in the current conversation; detailed method in `docs/research/paper_based_background_xray_flux_method.md`; reproducibility procedure in `outputs/paper_background_sensitivity/README.md`.

## Global Constraints

- Preserve the README's existing project overview and benchmark instructions.
- Describe project RXFI as the fractional excess `(F_peak - B) / B`; identify that Hong et al. describe peak-to-background ratio `F_peak / B`.
- Attribute the selected 24-hour method to Hong et al., Section 2.1, and link the NSF PAR paper PDF and the local method note.
- Describe the calculation script accurately: it requires the pinned NOAA catalog and 43 locked XRS L2 files, produces capped and uncapped event backgrounds, and writes fingerprinted outputs.
- State that existing generated targets still use NOAA `background_irrad`; this script does not regenerate forecast-window targets.
- Keep the legacy calendar-year flare-interval lookup caveat visible or directly linked from the README.

## Review Focus

- Readers could assume the research method already replaced the benchmark targets: explicitly distinguish the event-level analysis from existing target-generation inputs.
- Readers could confuse paper RXFI with project RXFI: show both formulas and identify the project's selected transform.
- Readers could expect a clean offline run to download data: document the local cache requirement for `--offline` and link online behavior to the run guide.
- Readers could treat NOAA's `background_irrad` as the paper estimator: say it remains a comparison field for the new calculation.
- A relative documentation link could be broken: verify each new README link resolves from repository root.

---

### Task 1: Explain the selected paper-based background and cite its source

**Files:**
- Modify: `README.md`

**Interfaces:**
- Cite: Hong et al., *Beyond Traditional Flare Forecasting: A Data-driven Labeling Approach for High-fidelity Predictions*, Section 2.1, linked to `https://par.nsf.gov/servlets/purl/10435332`.
- Link the detailed repository note `docs/research/paper_based_background_xray_flux_method.md` and comparison report `docs/research/paper_background_cap_sensitivity.md`.

- [x] **Step 1: Add README content distinguishing NOAA background from the selected estimator.** In the RXFI/background explanation, describe the preceding UTC 24-hour mean of valid one-minute GOES XRS-B samples, exclusion of other-flare intervals, quality screening, and the paper's start-flux upper cap. Explain that the reproducibility run also reports an uncapped sensitivity result.
- [x] **Step 2: State the provenance and transform conventions.** Cite the paper inline with author/title/section and PDF link. Show paper ratio `F_peak / B` separately from the project's selected fractional excess `(F_peak - B) / B`. State NOAA `background_irrad` is comparison-only for this analysis and existing generated targets have not yet been regenerated with the paper-based estimator.
- [x] **Step 3: Document script usage in a dedicated quick-start subsection.** Include environment activation, the repository-root command `python scripts/compute_paper_background_sensitivity.py --offline`, the required cache/catalog, the fingerprinted run output location, and a link to the run guide for online mode, custom paths, checksums, and overwrite behavior.
- [x] **Step 4: Surface the known legacy boundary caveat.** State briefly or link directly to the method/run note that the replay preserves the historical calendar-year catalog-interval lookup, which can omit a prior-year flare interval overlapping a window after New Year's.
- [x] **Step 5: Validate the documentation change.** Check new relative links resolve, confirm CLI flags against `--help`, and run `git diff --check`; do not run project tests for this documentation-only change.

**Checkable result:** A reader can identify the background method and source, distinguish it from NOAA's background field and the paper's RXFI formula, and reproduce the event-level outputs with the documented command without mistaking it for forecast-window target regeneration.
