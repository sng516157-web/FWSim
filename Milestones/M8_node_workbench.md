# M8 — Node Workbench (platform builder + control design)

> **Stage prompt for Kimi Code.** Read first:
> - `../docs/literature/flappy_flapsim_strengths.md` (S3, S4 — actuators/sensors/channels)
> - `../docs/architecture.md` — the data flow this milestone extends
> - `../docs/decisions/DR-002_ui_shell.md`, `DR-004_cad_derived_geometry.md`
>
> **Goal:** a Simulink/Simscape-style node layer for (a) assembling new robot
> platforms (CAD/URDF import + typed block hierarchy) and (b) building control
> pipelines (sensor → estimator → controller → mixer → actuator → plant), with
> everything compiling down to the SAME YAML/record formats — no second
> runtime, no shadow semantics.

## Design rules (from the user's rubric, applied to this milestone)

1. **Why should it exist:** the YAMLs are the truth but not a design surface;
   the node layer is the design surface. If a node doesn't map to a YAML
   block, it doesn't exist.
2. **How it integrates:** the graph compiles to vehicle/experiment/pipeline
   YAML and runs through `run_experiment`. One data flow, no side channels.
3. **How it's not confusing:** typed ports + hierarchy (not free wiring);
   the platform is a TREE (mechanical topology is a tree), the control
   pipeline is a DAG (signal flow). Two different graph shapes, not one
   free-for-all canvas.

## Key design decisions (drafted here, resolved in Task 0)

- **Platform builder = tree editor, not graph canvas.** Bodies → joints →
  wings form a kinematic tree; Simscape's lesson is that pretending it's a
  free graph confuses everyone. Tree editor + typed parameter panels + URDF
  import (M6.75 importer) as a node source.
- **Control pipeline = signal DAG.** Block library: sensor (IMU/motion-capture
  sim — S3 backlog lands here), estimator (complementary filter first),
  controller (PID, LQR from control/), channel mixer (flappy's
  wing-beat-modulation channels, S4), actuator (torque-driven from M3), plant
  (free_body plant), scope. Signals are typed (state / wrench / scalar-channel).
- **Runtime = dataflow executor** compiling the DAG to a callable used by the
  plant's control hook (new `control/pipeline.py`; the plant calls
  `pipeline(state, t) -> commands` each step). Batch and live sessions share it.

## Task 0 — Node schema + pipeline executor (`control/pipeline.py`)

YAML block: `pipeline: [{id, type, params, inputs: [{port, from}]}]`;
executor topologically sorts and evaluates. Blocks are typed with declared
input/output ports; connection errors fail loudly at load. Blocks: sensor_imu,
sensor_mocap, estimator_complementary, controller_pid, controller_lqr,
mixer_wingbeat_channels (S4), actuator_torque, scope. Tests: DAG ordering,
type checking, cycle rejection, equivalence of a hand-written pipeline with
the M6 closed-loop demo (same numbers).

## Task 1 — Canned block library + tests

The blocks above as documented, provenance-tagged library entries with unit
tests each (IMU noise model vs spec, complementary filter converges, PID
matches a reference response, mixer maps channels to documented kinematics
modulations).

## Task 2 — GUI node editor (`ui/nodes/`)

QGraphicsView-based editor (evaluate NodeGraphQt first — maintenance risk
noted; custom QGraphicsScene is the fallback and the survey-consistent
choice since we already own PySide6). Two canvases: platform tree, pipeline
DAG. Compiles to YAML; "run" runs it. Tested offscreen (graph ops only —
rendering is the human check).

## Task 3 — Platform builder (CAD → nodes → vehicle YAML)

Tree editor with URDF import as node-source (M6.75 importer), wing blocks
added by hand, export to vehicle YAML (verbose provenance format). The
AirPulse-style workflow: import URDF → attach wings → save → run.

## Task 4 — Closed-loop pipeline demo + validation

Pipeline-driven hover on the Level-2 free-body plant: IMU sim → complementary
estimator → PID attitude + LQR position → wing-beat mixer → actuator → plant.
Controller sees ESTIMATED state (S3), not truth. Report: response vs the
M6 truth-state loop (honest degradation table).

## Acceptance

```bash
pytest tests/ -q
fwmav-sim run experiments/m8_pipeline_hover.yaml   # pipeline-driven hover
fwmav-sim gui                                       # node editor: build + run
```

**Effort honesty:** this is the largest UI milestone yet (weeks, not days).
Task 0+1 are the physics-relevant core and land first; the canvas editor is
the separable front half.
