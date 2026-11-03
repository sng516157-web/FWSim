# Milestone 5 — Integrated Workbench

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — especially §3.1, §8.2, §9, §10 M5
> - `../FWMAV_Milestone_Action_Plan.md` — stack §1, **decision record DR-002 §1.3** (UI shell), conventions §3
> - M1–M4 milestone reports — everything the UI shows must already exist as a tested numerical capability
>
> Conflicts: the brief wins; stop and flag. Route A: pure Python + PySide6; no web stack, no JS toolchain.

**Goal:** a new user executes the complete reference workflow (brief §9 steps 1–11) without editing source code; UI and headless runs produce equivalent numerical outputs (brief §10 M5).
**Estimated effort:** 3–4 weeks.
**Prerequisite:** M4 accepted. **Do not start with a dashboard of disconnected controls** (brief §9) — the UI is a shell over working science, never the other way around.

---

## 5.1 Ground rules for this stage

- **Physics before interface (brief §3.1).** The UI adds zero new physics. Every number it displays comes from `experiment/record.py` outputs.
- Every demo plot is generated from the current model, never from stored presentation data (brief §10 M5 acceptance) — enforced by test, not by discipline.
- Headless/UI numerical equivalence is an automated acceptance test, not a manual check.
- The assumption/validity panel is a first-class view (brief §9), fed by the honesty hooks in the run records.

## 5.2 Locked stack for this stage (adds UI only)

**PySide6** (shell, forms, docking) · **pyqtgraph** (time-series) · **MeshCat in `QWebEngineView`** (3D) · everything from M0–M4 unchanged. Update action plan §1.1 with PySide6/pyqtgraph version pins when added (dependency protocol §5).

---

## 5.3 Tasks (in order)

### Task 1 — Application shell (`ui/app.py`, `ui/main_window.py`)

PySide6 main window with dockable panels. Single process, imports `fwmav_sim` directly — no server, no IPC. Entry point: `fwmav-sim gui`.

### Task 2 — Vehicle/tree inspector (`ui/inspector.py`)

Tree view of `VehicleDefinition`: bodies, joints, wings, frames. Selecting a node shows its parameters with units and provenance badges (`measured / literature / estimated / fitted` — colour-coded; `estimated` values visually distinct, brief §14.6).

### Task 3 — Parameter editor (`ui/parameter_editor.py`)

Forms **auto-generated from the Pydantic schema**: units and valid ranges shown, validation errors inline, provenance editable per parameter. Editing writes back to a YAML experiment definition — the same file the CLI consumes. No second config format.

### Task 4 — 3D view (`ui/scene3d.py`)

MeshCat in `QWebEngineView`: vehicle geometry, frame triads, wing surfaces, and the **CoM / aerodynamic-centre / lever-arm overlays** (brief §9 UI priorities) driven by `dynamics/mass_properties.py` and M3's lever-arm outputs. Force-vector overlays from per-term force outputs.

### Task 5 — Time-series view (`ui/timeseries.py`)

pyqtgraph plots reusing `analysis/` products: instantaneous per-term forces, cycle-averaged traces, state trajectories, residuals (M4). Live-update during a run is optional; correctness of displayed data is not.

### Task 6 — Run control + validity panel (`ui/run_panel.py`, `ui/validity_panel.py`)

Experiment selection, run/stop, progress, solver settings view; on completion the record opens for inspection. Validity panel lists assumption IDs in force for the current configuration (from the run record's validity section) plus regime warnings — e.g., quasi-steady limits, prescribed-vs-actuated mode, excluded physics (brief §3.5).

### Task 7 — Experiment save/load + workflow wiring

UI wraps `experiment/record.py` save/load. Wire brief §9 steps 1–11 as the tested reference workflow: select reference vehicle → inspect → configure → fixed-body run → inspect forces → enable abdomen → suspended test → compare (M4) → free-body run → save/reload → headless repeat.

### Task 8 — Packaging + quickstart

Clean-venv install check (`pip install .` then `fwmav-sim gui` and `fwmav-sim run`); README quickstart that walks the reference workflow end to end.

---

## 5.4 Test inventory

**Headless/UI equivalence (§11.3):** the same YAML executed via UI action and via `fwmav-sim run` ⇒ outputs equal within the documented tolerance rule from M3 (`tests/numerical/test_ui_headless_equivalence.py`).

**No stored presentation data:** a test regenerates each UI demo plot's data from a fresh run and compares against the displayed buffer (`tests/unit/test_demo_plots_live.py`).

**Integration:** the full §9 workflow scripted as an integration test (`tests/numerical/test_reference_workflow.py`), including step 11 — headless rerun reproducibility.

**Unit:** schema→form generation round trip; provenance editing validation.

## 5.5 Acceptance commands

```bash
pytest tests/ -q
fwmav-sim gui                                   # manual acceptance: full §9 workflow, no source edits
fwmav-sim run experiments/m3_free_body_hover.yaml   # headless equivalence spot-check
```

## 5.6 Milestone report

`docs/milestone_reports/M5.md`: brief §10 M5 checklist verbatim with evidence — new-user workflow walkthrough, headless/UI equivalence numbers, demo-plot freshness test result, clean-install log, screenshot set of inspector / 3D overlays / validity panel.

## 5.7 Explicit non-goals for this stage

Visual block-diagram controller editor (brief §7: programmatic composition suffices); photorealistic rendering (§4.2); any new physics, aerodynamics, or actuator models; multi-user or web deployment.
