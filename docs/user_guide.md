# FWSim User Guide

A 10-minute path from install to a hover force record, plus the four concepts
the whole tool is built around. New here? Start with `examples/` (tiny
runnable scripts) and `docs/architecture.md` (the one-page map).

## Concepts (read this first — everything else follows)

| Concept | What it is | Where it lives |
|---|---|---|
| **Vehicle** | The physical aircraft: bodies (mass/CoM/inertia), joints, wings (planform, hinge), aero coefficients — every value carries provenance (`measured` / `literature` / `estimated` / `fitted`). | `reference_vehicles/*.yaml` |
| **Experiment** | A run recipe: which vehicle, what wing kinematics, what air, what solver, what kind (`fixed_body` rig, `suspended_pendulum`, `free_reduced`, `free_body`, `closed_loop_reduced`). | `experiments/*.yaml` |
| **Record** | The saved output of a run: time series + config + vehicle copy + metadata (status, assumptions, warnings). Self-contained and reproducible. | `outputs/<timestamp>_<name>/` |
| **Validity** | The honesty layer: which assumptions (A-001…) were in force for a run and which regime warnings apply. Read it before trusting any number. | record metadata, GUI validity panel |

The golden path: **Load** an experiment → **Inspect/Edit** the vehicle and
kinematics → **Run** → **Results** (scope + 3D) → **Validity**.

## Install

```bash
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows
pip install -e .[dev,gui]
pytest tests -q                                          # sanity: all pass
```

## Your first run (headless)

```bash
fwmav-sim run experiments/m1_fixed_body_hover.yaml
```

This simulates the reference vehicle clamped on a force rig for 5 wingbeats
and writes a record to `outputs/`. Inspect it:

```bash
fwmav-sim report outputs/<record_dir>     # force figures
fwmav-sim view outputs/<record_dir>       # animated 3D in your browser
fwmav-sim rerun outputs/<record_dir>      # prove bit-identical reproducibility
```

Other useful commands:

```bash
fwmav-sim run experiments/m3_free_body_hover.yaml   # the MAV actually flies
fwmav-sim sensitivity experiments/m1_fixed_body_hover.yaml   # what matters most
fwmav-sim import-urdf robot.urdf -o my_vehicle.yaml          # CAD -> vehicle skeleton
fwmav-sim fit-sysid <fixed_body_exp> <dataset_dir> --columns FX FY FZ -o fit.yaml
```

## The GUI

```bash
fwmav-sim gui
```

- **Ribbon (top):** the golden path as five buttons; the current stage
  highlights.
- **Left dock:** `inspector` (vehicle tree with provenance badges),
  `parameters` (edit values — matrices editable, invalid entries flagged
  red immediately, ↺ resets a field), `records` (every run in `outputs/`;
  double-click to open, right-click to compare-overlay two runs).
- **Centre:** `3D` (vehicle/wings/forces; Play/scrub/speed after a run;
  opens a preview when you load an experiment) and `time series` (the scope:
  tick channels to overlay, drag the cursor to scrub the 3D frame, toggle
  the phase average).
- **Right dock:** experiment picker + run button, and the **validity** panel
  (assumptions in force, warnings — read before trusting numbers).
- **Bottom dock:** the **console** — every warning, error and traceback
  lands here. Nothing fails silently anymore.

### Interpreting the experiment kinds

| Kind | Analogy | Body moves? |
|---|---|---|
| `fixed_body` | force-balance rig | no (clamped) |
| `suspended_pendulum` | hanging from a string at a pivot | pendulum only |
| `free_reduced` | point-mass longitudinal model (Level 1) | yes (x, z, pitch) |
| `free_body` | full 6-DoF articulated flight (Level 2) | yes |
| `closed_loop_reduced` | Level-1 plant + LQR demo | yes |

## Provenance discipline (why every number has a tag)

Every physical parameter carries `source`: `measured` (hardware),
`literature` (cited), `estimated` (placeholder — treat results as
pipeline demonstrations), `fitted` (identified from data by sysid).
Records carry the same tags into their validity section. The rule of thumb:
**conclusions may only be drawn at the strength of the weakest provenance
in the chain.**

## Where things are

- Physics: `src/fwmav_sim/{kinematics,aero,dynamics,simulation}`
- Analysis: `src/fwmav_sim/{analysis,datasets}` (averaging, sensitivity, sysid, compare)
- Conventions (frozen): `docs/conventions/` · assumptions: `docs/assumptions.md` ·
  decisions: `docs/decisions/DR-*.md` · milestone evidence: `docs/milestone_reports/`
- Experiments: `experiments/` · vehicles: `reference_vehicles/` · outputs: `outputs/`
