# Milestone 6 — Usability & Physics Fidelity

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — especially §3.1 (physics before interface), §5.1, §5.8, §9
> - `../FWMAV_Milestone_Action_Plan.md` — stack §1, dependency protocol §5
> - M0–M5 milestone reports + P1–P4 post-M5 reports
>
> Conflicts: the brief wins; stop and flag. Route A: pure Python + PySide6; no web stack, no JS toolchain.

**Goal:** (a) the workbench stops failing silently and guides the standard loop the way MATLAB's toolstrip/console/scope do; (b) the aero core gains the differentiating unsteady terms with regime-appropriate, swappable coefficient sets; (c) fitting machinery closes the measure→fit→validate loop; (d) a closed-loop control demo runs end to end; (e) CI and a user guide make the project maintainable by others.
**Prerequisite:** M5 + P1–P4 accepted (baseline: 142 passed / 4 skipped).

**Out of scope (recorded, not silently dropped):** a fully validated vehicle from hardware rig data (needs the user's physical platform); wing flexibility (A-001 relaxation); pyvista 3D migration (browser-view option instead).

---

## 6.1 Ground rules for this stage

- Physics before interface (brief §3.1): UI tasks add zero new physics.
- Every new physics term is cited, provenance-tagged, and off by default; turning it on must be visible in the run record's validity section.
- Silent failure is a defect class: exceptions reachable by a user action must surface in the console dock (Task 1) — tested, not disciplined.
- Every panel change keeps headless/UI numerical equivalence (M5 acceptance, unchanged).

## 6.2 Locked stack for this stage

Everything from M0–M5 unchanged. New runtime dependencies: none. New dev dependencies: none (CI uses the existing `[dev]` extra).

---

## 6.3 Tasks (in order)

### Task 1 — Console dock + error surfacing (`ui/console.py`, app-wide hook)

Bottom dock, colour-coded log (info / warning / error), fed by: Python logging, Qt-slot exception hook (`sys.excepthook` + a `qInstallMessageHandler`), and run-worker failures (full traceback + failing state summary). Run failure ⇒ modal dialog AND console entry. Test: an action that raises produces a console entry and a non-empty error record (offscreen).

### Task 2 — Workflow ribbon (`ui/workflow.py`)

Toolstrip across the top: **Load → Inspect/Edit → Run → Results → Validity**, each a button that raises the matching panel; the current stage highlights automatically (experiment loaded / record loaded / edited-unsaved states). The strip replaces menu-discovery for the golden path.

### Task 3 — Scope panel (`ui/timeseries.py` rework)

Simulation-Data-Inspector style: channel tree with checkboxes (multiple overlays), linked crosshair cursor across all visible plots, and cursor↔3D linkage — clicking a time in a plot scrubs the 3D panel to that frame (`scene3d` frame seek). Cycle-average overlay toggle (analysis/averaging). Test: cursor linkage updates the 3D frame; checkbox state replots correctly (offscreen).

### Task 4 — Parameter editor rework (`ui/parameter_editor.py`)

Spreadsheet grid for Vector3/Matrix3 (inertia tensors editable), vehicle AND experiment trees editable side by side, immediate invalid-state styling (red border on schema violation, before save), provenance column, per-field reset-to-file. Test: matrix edit round-trips through YAML; invalid input is flagged inline without crashing.

### Task 5 — Records browser (`ui/records.py`)

Left dock listing `outputs/` newest-first: name, kind, OK/FAILED icon, timestamp. Double-click opens the record (time series + 3D + validity). Right-click "compare with…" overlays two records' matching channels in the scope. Test: browser discovers a written record and opens it (offscreen, monkeypatched render).

### Task 6 — Aero coefficient registry (`aero/registry.py`, schema)

Named coefficient sets as first-class data: `dickinson1999_robofly` (current), plus a documented higher-Re flat-plate set with literature citation. Vehicle YAML selects by name OR keeps inline coefficients (backward compatible). Run record validity section names the set and its regime. Test: registry loads, regime metadata present, inline coefficients still work.

### Task 7 — Unsteady terms: wake capture + clap-and-fling (`aero/unsteady.py`)

Optional add-on force terms, off by default, each citable (e.g. Berman & Wang 2007 for wake capture; Weiss-Fogh/Ellington lineage for clap-and-fling) and each gated by a named flag in the aero spec. Wake capture: impulse term near stroke reversal proportional to wake energy proxy from the previous half-stroke. Clap-and-fling: active only when opposing-wing gap falls below a threshold (X-wing/butterfly topology). Each term: analytic unit tests (sign, magnitude scaling, zero in its off-regime) + a fixed-body experiment showing its effect. Recorded in validity warnings when active.

### Task 8 — Control design loop (`control/`)

`control/trim.py`: hover trim solver for the free-body plant (least-squares over kinematic offsets: pitch offset, stroke plane, abdomen offset — driver residual = cycle-mean wrench). `control/linearize.py`: numeric A/B of the trimmed plant (reduced model has the analytic path; free body numeric, documented). `control/lqr.py`: scipy-based LQR over the linearization. Closed-loop experiment: trimmed hover + LQR on the reduced model, perturbed initial state — converges; test pins settling behaviour and control-signal sanity. This is a demo loop on the Level-1/2 models, NOT a flight-ready autopilot (recorded in the report).

### Task 9 — System-identification fitting (`analysis/sysid.py`)

Least-squares fit of the CL/CD fit-form coefficients to a measured force time series (M4 dataset format): given recorded wing kinematics + measured forces, fit coefficients via scipy least_squares with parameter bounds; write back a vehicle fragment with `source: fitted` provenance + the dataset reference. Proof on synthetic data: fit recovers the known coefficients used to generate the series (< 5% error on the informative ones; documented identifiability limits). Real-hardware fitting stays open (recorded).

### Task 10 — CI + changelog + user guide

`.github/workflows/ci.yml`: pip install .[dev,gui], ruff, full pytest on push/PR (ubuntu-latest; GUI tests offscreen). `CHANGELOG.md` seeded from the git history (M0–M5, P1–P4, M6). `docs/user_guide.md`: concepts page (vehicle / experiment / record / validity mental model) + 10-minute quickstart from install to hover, matching the actual GUI.

---

## 6.4 Test inventory

- Each task's tests above land in the matching tests/ subdirectory (unit / numerical / validation per existing conventions).
- Full suite must stay green with every task committed; M6 report carries the final counts.

## 6.5 Acceptance commands

```bash
pytest tests/ -q
fwmav-sim gui                       # manual: golden path via ribbon; error surfaces in console
fwmav-sim run experiments/m6_wake_capture.yaml
fwmav-sim run experiments/m6_closed_loop_hover.yaml
python -m fwmav_sim.analysis.sysid --help   # fitting CLI smoke
```
