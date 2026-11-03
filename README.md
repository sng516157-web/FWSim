# FlapSim — FWMAV Simulation and Control Workbench

A configurable flapping-wing micro aerial vehicle (FWMAV) flight-dynamics and
control workbench, with first-class support for morphing-body control mechanisms
such as an actuated, CoG-shifting abdomen.

This repository is under construction. The behavioural and scientific contract is
`FWMAV_Simulator_Project_Brief.md`; the build order is
`FWMAV_Milestone_Action_Plan.md`; per-stage prompts live in `Milestones/`.

## Conventions (frozen in Milestone 0)

- All units are SI internally.
- Coordinate frames, angle signs and rotation order are defined in
  `docs/conventions/frames.md` and are frozen — later code must not redefine them.
- Every physical symbol in code matches `docs/conventions/notation.md`.
- Every physical parameter carries schema-enforced provenance
  (`src/fwmav_sim/vehicle/schema.py`).
- Modelling assumptions live in `docs/assumptions.md`; code relying on an
  assumption cites its ID (e.g. `A-003`) in its docstring.

## Install (Windows, Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .[dev]
```

## Quick checks

```bash
ruff check src tests
pytest tests/unit -q
pytest --collect-only -q
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Modelling specification and test plan | in progress |
| M1 | Fixed-body aerodynamic vertical slice | not started |
| M2 | Two-wing vehicle + analytical abdomen model | not started |
| M3 | Nonlinear articulated free-body simulation | not started |
| M4 | Validation and experiment comparison | not started |
| M5 | Integrated workbench | not started |
