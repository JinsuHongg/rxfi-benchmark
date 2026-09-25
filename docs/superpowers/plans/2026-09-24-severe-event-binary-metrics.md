# Severe-Event Binary Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible severe M+/X+ binary evaluation artifacts from completed predictions without retraining or changing existing results.

**Architecture:** A stdlib/NumPy/sklearn metric module will own class conversion, guarded binary metrics, thresholds, and ranking metrics. A CLI evaluator will schema-check the existing QR CSVs and corrected classification matrices, create new CSV artifacts beneath `outputs/severe_event_binary_metrics/`, then generate the report from those outputs.

**Tech Stack:** Python 3.11, csv/json/pathlib, NumPy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-severe-event-binary-metrics-design.md`

## Global Constraints

- Do not retrain or load models; do not alter source predictions, split files, checkpoints, target generation, or prior result files.
- Use validation q50 labels/scores only for threshold selection; never use test labels/metrics/sweeps to select a threshold.
- Only `max_peak_flux` q50_raw receives M1/X1 physical thresholds.
- Verify corrected classification matrix labels against each saved class mapping; non-test classification metrics and classification score metrics remain unavailable.
- Store new generated outputs only in `outputs/severe_event_binary_metrics/`.

## Review Focus

- A no-positive or no-negative population yields explicit `NaN` ratios, not an exception or invented zero.
- A tie in validation TSS resolves by positive F1, then POD, then the higher threshold.
- Test-label changes cannot alter a selected threshold.
- Classification matrices with reordered or mismatched labels are rejected rather than silently collapsed.
- Physical thresholds are never applied to cumulative flux or RXFI.

---

### Task 1: Binary metric and threshold module

**Files:**
- Create: `scripts/severe_event_metrics.py`
- Create: `tests/test_severe_event_metrics.py`

**Interfaces:**
- Produces: `event_binary_labels`, `binary_metrics_from_counts`, `binary_metrics`, `ranking_metrics`, `threshold_curve`, and `select_validation_threshold`.

- [ ] **Step 1: Write failing metric/mapping/threshold tests**

Include perfect, all-negative, all-positive, known `TP=3, FP=1, TN=4, FN=2` matrix, M+/X+ class conversion, direct class collapse, validation-only selection, and test-label independence.

- [ ] **Step 2: Run the focused tests to verify red**

Run: `pytest tests/test_severe_event_metrics.py -q`

Expected: FAIL because `scripts.severe_event_metrics` is absent.

- [ ] **Step 3: Implement the minimal reusable metric module**

Implement guarded count-derived metrics, sklearn ranking calls, exact unique-score threshold curves, and the stipulated tie ordering.

- [ ] **Step 4: Run focused tests to verify green**

Run: `pytest tests/test_severe_event_metrics.py -q`

Expected: PASS.

### Task 2: Artifact evaluator and reproducible CSV outputs

**Files:**
- Create: `scripts/evaluate_severe_event_binary_metrics.py`
- Modify: `tests/test_severe_event_metrics.py`

**Interfaces:**
- Consumes: Task 1 public functions and saved QR/matrix artifacts.
- Produces: all machine-readable output tables, threshold curves, and availability manifest under `outputs/severe_event_binary_metrics/`.

- [ ] **Step 1: Write failing evaluator contract tests**

Use temporary mini CSVs/matrices to require schema validation, verified matrix labels, frozen validation threshold use across test, and max-peak-only physical eligibility.

- [ ] **Step 2: Run focused tests to verify red**

Run: `pytest tests/test_severe_event_metrics.py -q`

Expected: FAIL because the evaluator helpers are absent.

- [ ] **Step 3: Implement the evaluator CLI**

Load exact requested sources, emit classification TEST rows and unavailable manifest entries, select QR thresholds only from validation, and write standard column order for metrics and curves.

- [ ] **Step 4: Run the evaluator and focused tests**

Run: `python scripts/evaluate_severe_event_binary_metrics.py && pytest tests/test_severe_event_metrics.py -q`

Expected: output directory populated and tests PASS.

### Task 3: Research documentation and verification

**Files:**
- Create: `docs/research/severe_event_binary_metrics.md`
- Modify: `docs/research/current_state.md`

- [ ] **Step 1: Generate compact result summaries from the new CSVs**

Run: `python scripts/evaluate_severe_event_binary_metrics.py`

Expected: deterministic new-result artifacts only.

- [ ] **Step 2: Document protocol, sources, results, and limitations**

Include requested Project 1, Project 2 classification, and Project 2 QR TEST tables; list thresholds; make ranking-vs-decision distinction and unavailable classification scores explicit.

- [ ] **Step 3: Run full regression verification**

Run: `pytest -q && python scripts/evaluate_severe_event_binary_metrics.py`

Expected: all tests pass and evaluator completes without modifying source artifacts.
