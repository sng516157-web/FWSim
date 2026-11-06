# M9 — Workbench redesign: mechanical tree + control DAG + live loop

> **Status:** design proposal, awaiting user sign-off before implementation.
> **Supersedes:** the M8 "nodes" tab as shipped (pipeline canvas + read-only
> platform tree) — the concepts survive, the UX does not.
> **Research basis:** Webots (scene tree, PROTO), Simulink/Simscape
> (library browser, physical networks, Dashboard live tuning), XFLR5 /
> PlanformCreator2 (wing planform vector editing). Notes inline below.

## 1. What the user actually asked for

The workflow loop, verbatim:

    build robot (primitives or CAD) → build controller (nodes) →
    run simulation LIVE → view results live + recorded → tune controller → repeat

The current GUI fails this loop in three ways:
1. The robot is a YAML you edit in a table — there is no *building*.
2. "Run" is batch-then-inspect; live exists but is a side-channel, not the
   centre of the loop.
3. Tuning means editing YAML and re-running — there is no live tuning.

## 2. Lessons from the reference tools (searched, distilled)

**Webots** ([scene tree docs](https://cyberbotics.com/doc/reference/using-proto-nodes-with-the-scene-tree)):
- The robot is a **scene tree**: Robot → children (Solid, Joint, Device…).
  Adding a node = pick from a typed node list, it lands in the tree.
- **PROTO** = parameterized reusable subtree (a "wing" with span/chord
  fields is a PROTO). Our vehicle YAML sections play this role.
- The 3D view is *always* the simulation; there is no batch mode in the UI.
  "Run" is a play button; the scene updates live.

**Simulink/Simscape** ([Simscape overview](https://www.cesimulations.com/simscape-multibody/)):
- **Two domains, two editors.** Simscape Multibody models mechanics as a
  *physical network* (bidirectional ports — energy flows both ways);
  Simulink models control as a *signal DAG* (directed ports). They meet at
  actuator/sensor blocks. This is exactly the split the user asked for.
- Library browser on the left, canvas in the middle, block parameters in a
  dialog on double-click.
- **Dashboard blocks** ([MathWorks docs](https://ww2.mathworks.cn/help/simulink/ug/connect-dashboard-blocks.html)):
  sliders/knobs bound to block parameters, writable *while the sim runs*.
  This is the "tune controller live" ask.

**XFLR5 / PlanformCreator2** ([PC2 repo](https://github.com/jxjo/PlanformCreator2)):
- Wing geometry is authored as a **planform outline**: leading-edge and
  trailing-edge polylines in the plan view, dragged point-by-point; chord
  distribution, area, MAC, and paneling derive automatically and update
  on every edit. This is the "vector drawing" the user described.

## 3. The redesign

### 3.1 Two node domains (answers question 1)

**Mechanical domain = a TREE editor** (Webots scene-tree style), not a
canvas. Mechanical topology *is* a tree; a free canvas only adds confusion
(M8 design rule 3, confirmed by how Webots/Simscape both do it).

Node types (primitives — no CAD required):
- `body` — box / cylinder / point mass primitive; mass properties derived
  from dimensions × density, or typed in directly.
- `joint` — revolute / prismatic / fixed, typed axis + limits.
- `wing` — the special node, see §3.3.
- `cad_subtree` — a URDF import grafted in as a subtree (existing importer).

Each node selected in the tree shows its parameters in a property panel
(right side) and its effect in the 3D preview immediately.

**Control domain = the signal DAG** (existing M8 pipeline executor +
canvas — this part survives). Sensor/estimator/controller/mixer/actuator
blocks with typed ports.

**The boundary between them is explicit and typed:** the mechanical tree
compiles to a plant; the plant exposes typed ports (state out, channels
in). The DAG connects to those ports. One boundary, no side channels —
same rule as M8, now visible in the UI.

### 3.2 The window, rebuilt around the loop

```
┌────────────┬──────────────────────────────┬────────────────┐
│  library   │        canvas / 3D           │   properties   │
│  (palette) │                              │   (selected    │
│            │  tabs: [platform] [control]  │    node)       │
│  bodies    │         [3D live] [scopes]   │                │
│  joints    │                              │  dashboard:    │
│  wings     │  transport bar:              │  live-tunable  │
│  sensors   │  ▶ ⏸ ⏹  speed  ● record     │  params        │
│  ctrl blk  │                              │                │
└────────────┴──────────────────────────────┴────────────────┘
```

- **Left: library palette** — the node catalog (mechanical primitives +
  control blocks), drag or double-click to add. Simulink's library browser.
- **Centre: tabs.** *platform* = the mechanical tree + live 3D preview;
  *control* = the DAG canvas; *3D live* + *scopes* = the running sim.
- **Bottom of centre: transport bar** — play/pause/stop, speed, record
  toggle. Live is the default mode (Webots lesson).
- **Right: properties + dashboard.** Parameters of the selected node;
  parameters flagged as tunable appear as sliders/spinboxes that write
  through to the *running* sim (Simulink Dashboard lesson).

### 3.3 The wing node (vector-drawn planform)

Schema change first (this is the enabler): `planform` gains a `custom`
variant:

```yaml
planform: custom
planform_stations:   # spanwise stations, root -> tip
  - {y: 0.00, chord: 0.183, le_offset: 0.000}   # le_offset: LE x at station
  - {y: 0.15, chord: 0.183, le_offset: 0.000}
  - {y: 0.30, chord: 0.020, le_offset: 0.060}
```

The editor: a 2D canvas showing the half-wing in plan view; the user drags
leading-edge and trailing-edge points (PlanformCreator2 style). Every edit
recomputes chord(y), area, MAC, and the blade-element discretization
preview live. Wing mass (DR-006 thin-plate) and the pitch-axis line derive
from the same outline.

Consequence for the engine: `discretize()` must sample chord from stations
instead of the rectangular/trapezoidal/elliptical formulas (one new code
path, behind `planform: custom`).

### 3.4 The live loop (answers question 3)

- **Live is the default.** "Run" from any editor compiles (tree → vehicle
  YAML in memory, DAG → pipeline) and starts a LiveSession; 3D + scopes
  update in real time; the recorder is always on (stop = save record).
- **Live tuning.** Dashboard sliders write into the running pipeline's
  block params (blocks need a `set_param(name, value)` hook — small change
  to the block factories) and into mechanical parameters that are safe to
  change live (kinematics amplitude/frequency; not topology).
- **Everything is a record.** A live session saved is the same record
  format as a batch run (M6.75 SessionRecorder already does this — the
  redesign makes it the *only* path, removing the batch/live split).

## 4. What survives / what changes

Survives unchanged: physics (aero, dynamics, runners), record format,
pipeline executor, block library, platform builder core, URDF import,
MeshCat 3D, validity panel.

Rewritten: `ui/main_window.py` shell (palette + tabs + transport +
dashboard layout), the M8 "nodes" tab (becomes the two domain editors),
`ParameterEditor` (becomes the property panel + dashboard).

New: wing planform editor canvas, `planform: custom` schema + discretize
path, dashboard write-through into live sessions, library palette.

## 5. Proposed task breakdown

- **T0 schema:** `planform: custom` + stations; discretize from stations;
  tests (area/MAC against analytic shapes).
- **T1 wing planform editor:** 2D vector canvas (LE/TE drag points), live
  chord/area/MAC readout, round-trips to YAML.
- **T2 mechanical tree editor:** primitives (body/joint/wing/cad_subtree),
  add/remove/reparent, property panel, live 3D preview.
- **T3 shell rebuild:** palette + tabs + transport bar + dashboard;
  live-first run loop; record-on-stop.
- **T4 live tuning:** `set_param` on blocks; dashboard bindings; tune a
  running hover PID from sliders, on screen.
- **T5 Qt 3D migration:** Qt3D-based scene view alongside MeshCat; parity
  checklist (grid, wings, force arrows, replay scrub, live stream); switch
  default, then remove MeshCat.
- **T6 migration:** port existing experiments/vehicles; CLI batch path
  stays untouched; update user guide.

## 6. Decisions (user, 2026-11-06)

1. **Batch CLI stays as-is** for headless/CI; the GUI goes live-first.
2. **Wing editor = plan-view outline only**; twist stays in kinematics
   profiles.
3. **3D renderer moves to Qt 3D** (replacing MeshCat). Accepted as a
   larger, riskier task: nicer integration (no browser, no localhost
   server, no fullscreen glitches) at the cost of a rewrite of
   `viz/` and `ui/scene3d.py`. Migration plan: build the Qt 3D view
   alongside MeshCat, switch the default when parity is reached
   (grid, wings, force arrows, replay scrubbing, live streaming),
   then remove MeshCat.
