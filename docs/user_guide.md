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
fwmav-sim live experiments/m3_free_body_hover.yaml  # live dynamics streaming (not a replay)
fwmav-sim sensitivity experiments/m1_fixed_body_hover.yaml   # what matters most
fwmav-sim import-urdf robot.urdf -o my_vehicle.yaml          # CAD -> vehicle skeleton
fwmav-sim fit-sysid <fixed_body_exp> <dataset_dir> --columns FX FY FZ -o fit.yaml
```

## The GUI

```bash
fwmav-sim gui            # M9 live-first workbench (default)
fwmav-sim gui --legacy   # the M5-M8 batch-oriented window
```

**The M9 workbench** follows the loop *build robot → build controller → run
live → view + record → tune*:

- **Left dock — library:** the node palette. Mechanical primitives (box /
  cylinder / point bodies, joints, wings, CAD subtrees) belong to the
  *platform* tab; control blocks (sensors, estimators, controllers, mixers,
  actuators, scopes) belong to the *control* tab. Double-click to add.
- **Centre tabs:**
  - *platform* — the mechanical tree plus the **wing planform editor**
    (drag the leading/trailing edge points; area, MAC and the blade-element
    discretization derive live from the outline).
  - *control* — the signal DAG canvas. Drag from a blue output port to an
    orange input port; connections are type-checked on release and invalid
    ones are rejected with the reason.
  - *3D live* — the Qt-native OpenGL scene (orbit: drag; zoom: wheel).
  - *scopes* — time series with cursor scrubbing.
- **Transport bar (bottom):** ▶ play runs the sim *live* (speed selector),
  ⏹ stop saves a run record identical in schema to a batch run (recording
  is on by default — ● toggles it).
- **Right dock — dashboard:** sliders for the live-tunable parameters of
  the running controller (PID gains, mixer limits). Moving a slider retunes
  the *running* simulation — this is how you tune a controller while it
  flies.

**The legacy window** (`--legacy`) keeps the batch workflow: ribbon,
inspector, parameter editor, records browser, validity panel, console.

### Building a platform from primitives (no CAD needed)

```python
from fwmav_sim.vehicle.platform_builder import PlatformBuilder
from fwmav_sim.ui.nodes.planform_canvas import PlanformOutline

pb = PlatformBuilder("my_mav")
pb.add_body_primitive("thorax", "box", {"lx": 0.04, "ly": 0.02, "lz": 0.02},
                      density=500.0)              # mass + inertia derived
outline = PlanformOutline()                       # or draw it in the GUI
outline.insert_station(0.15)                      # add a mid-span station
outline.move_point("te", 1, 0.02, 0.15)           # taper the tip
pb.add_wing_from_outline("wing_right", "right", outline,
                         hinge_location=[0.0, 0.01, -0.005])
pb.add_wing_from_outline("wing_left", "left", outline,
                         hinge_location=[0.0, -0.01, -0.005])
pb.save_vehicle("reference_vehicles/my_mav.yaml")  # validated before writing
```

From CAD instead: `PlatformBuilder.from_urdf("robot.urdf")` imports bodies +
joints (revolute/fixed, coordinate-axis-aligned); add wings by hand as above.

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
