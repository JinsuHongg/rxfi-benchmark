# Scripts Topic Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Move all 33 scripts into approved topic directories, document every file in a README within each scripts directory, and update callers to the new paths.

**Architecture:** Treat the approved design in `docs/superpowers/specs/2026-10-13-scripts-topic-organization-design.md` as the canonical move map. Move implementation files without changing their behavior, then repair script-root calculations, imports, and repository references. The `scripts/README.md` index and each nested README will describe the purpose of its directory and the role of every file there.

**Tech Stack:** Python 3.11, Bash/Slurm launchers, pytest, Markdown.

**Spec:** `docs/superpowers/specs/2026-10-13-scripts-topic-organization-design.md`.

## Global Constraints

- Use the destination mapping in the approved design spec; account for every current script file exactly once.
- Do not retain compatibility wrappers at the old script paths.
- Preserve program behavior and repository-root data/config/output locations.
- Update all live invocations, imports, and documentation references to the new paths. Historical plan text may retain old paths when it describes the original plan.
- Create `scripts/README.md` and a README in every directory below `scripts/`, including nested Python package, project, and launcher directories.
- Every README must briefly state the directory's purpose and describe every file in that directory; include invocation examples where useful.
- Do not change configuration files, datasets, generated outputs, or model code.

## Review Focus

- A moved CLI could compute the wrong repository root: assert the `ROOT` constant in nested Project 2 and Project 3 modules resolves to the checkout root.
- Colocated packages could stop importing: test paper-background CLI/package imports and severe-event analysis imports from both direct execution and pytest module imports.
- Shell/Slurm jobs could still call stale paths: syntax-check launchers and search for old script paths after the move.
- A README could omit a moved file or describe only the folder: test that each directory README mentions every basename and manually review descriptions for accuracy.
- Existing executable bits could be lost during moves: compare each moved shell/Slurm file's mode with its Git baseline in the layout test.

---

### Task 1: Move scripts and preserve imports and repository-root behavior

**Files:**
- Move all 33 files according to `docs/superpowers/specs/2026-10-13-scripts-topic-organization-design.md`.
- Modify: moved Python scripts that calculate `ROOT`, modify `sys.path`, or import colocated helpers.
- Modify: `tests/test_paper_background_sensitivity.py`, `tests/test_severe_event_metrics.py`, `tests/test_project3_export_contract.py`, `tests/test_flare_class_band.py`.
- Test: `tests/test_script_layout.py` (new).

**Interfaces:**
- Existing CLI arguments and calculation behavior remain unchanged.
- New module paths follow the destination table in the design spec. In particular, the paper-background entry point is `scripts/background/compute_paper_background_sensitivity.py`, and its package is `scripts/background/paper_background/`.
- Repository root is the directory above `scripts/`, regardless of the moved script's nesting depth.

- [x] **Step 1: Add failing structure/import tests.** Assert every mapped file exists at its destination, no moved file remains at its old path, imports for the paper-background, severe-event, and Project 3 modules resolve at their new paths, nested Project 2/Project 3 `ROOT` values resolve to the repository root, and moved shell/Slurm files preserve their Git executable mode.
- [x] **Step 2: Run the focused tests and confirm the expected RED.** Run `pytest -q tests/test_script_layout.py`; it should fail because the destination files/READMEs are not present yet.
- [x] **Step 3: Move all script and package files according to the approved map.** Preserve executable bits on shell and Slurm files.
- [x] **Step 4: Update repository-root resolution and imports.** Ensure nested files use the repository root for data/config/output paths; repair background package imports and severe-event helper imports for both direct CLI and test module contexts.
- [x] **Step 5: Run layout and import tests.** Run `pytest -q tests/test_script_layout.py tests/test_paper_background_sensitivity.py tests/test_severe_event_metrics.py tests/test_project3_export_contract.py tests/test_flare_class_band.py` and confirm all pass.

### Task 2: Add file-level README documentation and update repository callers

**Files:**
- Create: `scripts/README.md` and README files in every directory under `scripts/`.
- Modify: root `README.md`, relevant `docs/research/*.md`, shell/Slurm launcher files, and any remaining tests or scripts that refer to old paths.
- Test: `tests/test_script_layout.py`.

**Interfaces:**
- All run commands use the new paths and are launched from the repository root unless explicitly stated otherwise.
- All scripts remain independently invocable through their documented command.

- [x] **Step 1: Extend layout tests to require each README to mention every file in that directory.** Check direct child files, including Python modules and launchers, and nested directories have their own README.
- [x] **Step 2: Write `scripts/README.md` as the topic index.** Link to each topic directory and summarize what belongs there.
- [x] **Step 3: Write each topic/subdirectory README.** Briefly explain the folder purpose and every contained file; include representative repository-root commands for runnable tools and launchers.
- [x] **Step 4: Update current references.** Replace old script paths in root README, research notes, tests, and shell/Slurm launchers with their new paths; leave only intentional historical plan references.
- [x] **Step 5: Run focused verification.** Run the layout test, targeted existing tests, `--help` on relocated CLIs, `bash -n` on every `.sh` and `.sbatch`, and `git diff --check`.

### Task 3: Audit the repository-wide move and run the available test suite

**Files:**
- Modify any remaining live references found by the audit.
- Modify: `tests/test_script_layout.py` only if the audit reveals an untested layout invariant.

- [x] **Step 1: Search for stale live references.** Search Markdown, Python, shell, Slurm, YAML, and config files for every old path from the design's move table. Classify historical plan references separately.
- [x] **Step 2: Verify all destination files and READMEs.** Confirm exactly 33 original script files were moved, no unplanned script was added or removed, and every scripts directory has its README.
- [x] **Step 3: Run the full pytest suite.** Record collection/runtime failures and distinguish known environment failures from relocation regressions.
- [x] **Step 4: Inspect the final diff and status.** Confirm only script relocations, path/import/doc updates, README files, and layout tests changed; report any limitation before considering the task complete.

**Checkable result:** Every script is in its approved topic directory, every directory under `scripts/` has a README describing the purpose and each file, all live entry points use the new paths, and focused runtime checks pass.


## Execution notes (2026-10-13)

- Relocated all 33 mapped script files and repaired repository-root constants and cross-folder imports.
- Added `scripts/README.md` and a README in every topic and nested scripts directory; the layout test checks every direct file is named.
- Updated current README, research-note, output-guide, test, shell, and Slurm references. Old paths remain only in historical planning documents and this plan's original move map.
- Focused relocation suite: 44 passed, 8 subtests passed. Launcher syntax checks and `git diff --check` passed.
- Full `pytest -q` collection is blocked only by the environment's `wandb` generated Protobuf 6.32.1 versus runtime 6.31.1 mismatch in `tests/test_flare_class_band.py`. The netCDF-dependent refinement script cannot show `--help` in this environment because `netCDF4` is not installed. Analysis scripts that are batch-only have no `--help` CLI.
