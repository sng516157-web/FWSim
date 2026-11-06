# FWSim Architecture (one-pager)

Everything in this repo hangs off **four concepts** and **one data flow**.
If you understand this page, you can navigate the whole codebase.

## The four concepts

| Concept | What it is | Files | Why it exists |
|---|---|---|---|
| **Vehicle** | The physical aircraft: bodies (mass/CoM/inertia), joints, wings, aero coefficients — every value with provenance | `reference_vehicles/*.yaml` → `vehicle/schema.py` | one honest source of what the machine is |
| **Experiment** | A run recipe: vehicle + kinematics + environment + solver + kind | `experiments/*.yaml` → `experiment/config.py` | a run must be reproducible from a file, not from memory |
| **Record** | The saved output: TimeSeries + config copy + vehicle copy + metadata | `outputs/<ts>_<name>/` via `experiment/record.py` | self-contained, rerun-able evidence |
| **Validity** | Which assumptions were in force + regime warnings | record metadata, assumption register `docs/assumptions.md` | you may only trust a number at the strength of its weakest link |

## The data flow

```
experiment YAML ──load──► dispatch_experiment ──► TimeSeries ──► record (disk)
                               │                        │
                 vehicle YAML ─┘                        ├─► CLI (report/view/compare)
                                                      └─► GUI panels (scope/3D/validity)
```

- **Headless and GUI are the same path.** `simulation/run.py:run_experiment`
  is the only entry point; the GUI's run button calls it. Equivalence is
  tested bit-identically.
- **Physics never reads the GUI, records, or the file system** (records read
  *from* physics, never the reverse).
- Every generalized force has exactly one producer (aero model, actuator,
  wing inertia, damping) and one consumer (the dynamics solver).

## Module map (src/fwmav_sim)

| Layer | Package | Job |
|---|---|---|
| Conventions | `core/` | frames, transforms, TimeSeries — frozen math |
| Motion | `kinematics/` | prescribed wing chains, blade elements (analytic rates) |
| Forces | `aero/` | quasi-steady model, registry of coefficient sets, unsteady terms, abdomen plate |
| Dynamics | `dynamics/` | mass properties, spatial articulated model, reduced longitudinal model, wing inertia, cycle averaging, aero damping |
| Experiments | `simulation/` | runners per experiment kind + `run.py` dispatch + records |
| Analysis | `analysis/` `datasets/` | averaging, sensitivity, compare, sysid, datasets |
| Interfaces | `control/` `vehicle/urdf_import.py` `vehicle/compact.py` `vehicle/platform_builder.py` | trim/LQR/closed loop, **pipeline executor + block library (M8)**, CAD import, compact authoring, platform builder (M9: primitives + outline wings) |
| Shell | `ui/` (incl. `ui/workbench.py`, `ui/nodes/`) `viz/` (incl. `gl_scene`, `gl_frames`, `gl_live`) | GUI panels, node canvases, Qt-native GL scene (M9) + MeshCat (legacy) — **zero physics** |

## Design rules (the three questions, applied)

1. **Why should it exist?** — one sentence, written in the module docstring.
   If a module can't state its job in one sentence, it's two modules.
   *Example:* `aero/registry.py` — "coefficient sets are first-class named
   data with regime metadata" — that sentence is the whole justification.
2. **How does it integrate?** — at exactly one boundary, named in the
   docstring. *Example:* `vehicle/compact.py` integrates only at
   `load_vehicle`/`load_experiment`; nothing downstream knows it exists.
3. **How is it not confusing?** — the sensible default is the visible path;
   machinery is opt-in. *Examples:* wings are massless unless you set
   `mass`; unsteady aero terms are off unless you enable them; the GUI opens
   in basic mode.

## Reading order for a newcomer

1. `examples/fixed_body_hover.py` (10 minutes of running code)
2. this page
3. `docs/user_guide.md` (concepts + GUI tour)
4. `docs/assumptions.md` (what the physics does NOT do)
5. only then: `src/fwmav_sim/aero/quasi_steady.py` and
   `src/fwmav_sim/dynamics/` — the physics core
