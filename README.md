# FWSim — FWMAV Simulation and Control Workbench

A configurable flapping-wing micro aerial vehicle (FWMAV) flight-dynamics and
control workbench, with first-class support for morphing-body control mechanisms
such as an actuated, CoG-shifting abdomen.

This repository is under construction. The behavioural and scientific contract is
`FWMAV_Simulator_Project_Brief.md`; the build order is
`FWMAV_Milestone_Action_Plan.md`; per-stage prompts live in `Milestones/`.

**New here?** Run `python examples/fixed_body_hover.py`, then read
`docs/architecture.md` (the one-page map) and `docs/user_guide.md`.

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

## Quickstart (reference workflow, brief §9)

```bash
# headless
fwmav-sim run experiments/m1_fixed_body_hover.yaml     # fixed-body wing forces
fwmav-sim report outputs/<run_dir>                      # regenerate figures
fwmav-sim run experiments/m2_actuated_pendulum.yaml    # suspended abdomen test
fwmav-sim compare outputs/<run_dir> datasets/synthetic/hover_case \
    --sim-channel phi_p                                # dataset comparison (M4)
fwmav-sim run experiments/m3_free_body_hover.yaml      # free-body 6-DoF
fwmav-sim rerun outputs/<run_dir>                      # headless repeatability
fwmav-sim view outputs/<run_dir>                       # live 3D (MeshCat)

# workbench (requires the gui extra: pip install -e .[gui])
fwmav-sim gui
```

In the GUI: file → open experiment… → inspector/parameter tabs on the left,
run/validity on the right, 3D and time-series in the centre. Everything shown
comes from run records — no stored demo data.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Modelling specification and test plan | accepted (report: `docs/milestone_reports/M0.md`) |
| M1 | Fixed-body aerodynamic vertical slice | accepted (report: `docs/milestone_reports/M1.md`) |
| M2 | Two-wing vehicle + analytical abdomen model | accepted (report: `docs/milestone_reports/M2.md`) |
| M3 | Nonlinear articulated free-body simulation | accepted (report: `docs/milestone_reports/M3.md`) |
| M4 | Validation and experiment comparison | accepted (report: `docs/milestone_reports/M4.md`) |
| M5 | Integrated workbench | accepted (report: `docs/milestone_reports/M5.md`) |
| M6 | Usability & physics fidelity | accepted (report: `docs/milestone_reports/M6.md`) |
| M7 | Fidelity ladder & learning interfaces | planned (`Milestones/M7_fidelity_ladder_and_interfaces.md`) |
