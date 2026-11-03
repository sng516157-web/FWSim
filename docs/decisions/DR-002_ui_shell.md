# DR-002 — UI shell: PySide6 desktop

**Status:** confirmed 2026-11-03 (Route A). **Milestone of effect:** M5 only.

## Decision

PySide6 desktop shell; time-series via pyqtgraph; 3D via MeshCat embedded in
`QWebEngineView`; parameter editor auto-generated from the Pydantic schema
(`src/fwmav_sim/vehicle/schema.py`).

## Rationale

- Single Python process shares objects with the sim core, making the M5
  headless/UI equivalence test trivial.
- Windows-native, no JavaScript toolchain; MeshCat is already the M1–M3 3D
  viewer (action plan §1.1).

## Rejected

- FastAPI + web frontend: duplicates state across a process boundary, heavier.
- Dashboard-first UI: explicitly forbidden by brief §9.

## Revisit trigger

Steven requests a browser-based workbench before M5 starts (action plan §6,
open item 2). Affects M5 only, not M0–M4.

Full record: `FWMAV_Milestone_Action_Plan.md` §1.3.
