# Changelog

All notable changes to FWSim, newest first. The project follows the
milestone structure of `FWMAV_Milestone_Action_Plan.md`; post-roadmap items
are numbered P1+; M6 is the usability/fidelity milestone.

## M6.75 — Unified Base Constraint & Platform Hardening (2026-11-06)

- **Live dynamics streaming** (`fwmav-sim live <experiment>`, GUI "live"
  button): the free-body plant integrates continuously and streams to
  MeshCat in slow motion — not a record replay. One physics path
  (`build_free_body_plant` shared with `run`). **Sessions record by default:**
  every solver step is captured via `SessionRecorder`; stopping saves a full
  run record (identical schema to batch runs), which the GUI then opens.
- **DR-007: base constraint as experiment data.** `base: {mode:
  fixed|free|pivot, release_time}` on every experiment; kinds become
  backward-compatible aliases (overrides recorded in the record). Launch
  transient (clamp → release) is now expressible
  (`experiments/m6p75_launch_transient.yaml`; frozen pre-release, continuous
  release — tested).
- Free-body records gain per-term per-wing force channels (rig schema).
- `dickson2006_hummingbird` registry set (Dickson et al. 2006 AIAA-2006-34,
  the flappy fit).
- `fwmav-sim benchmark` (runtime accounting) and `fwmav-sim ablate`
  (per-block on/off force tables).

## M6.5 — Clarity Pass (2026-11-05)

- Comparative analysis (`docs/literature/flappy_flapsim_strengths.md`);
  repo cleanup; `examples/` teaching scripts; compact vehicle authoring
  (`format: compact-v1`); `docs/architecture.md`; GUI basic mode.

## M6.5-pre — rename (2026-11-05)

- Project renamed **FlapSim → FWSim** (name collision with arXiv:2610.40902).

## M6 — Usability & Physics Fidelity (2026-11-05, in progress)

- **Workbench (MATLAB-inspired UX):** console dock with global error
  surfacing (no more silent failures), workflow ribbon (Load → Inspect →
  Run → Results → Validity), scope panel with checkbox channel overlays +
  cursor↔3D frame linkage + phase-average overlay, parameter editor rework
  (editable inertia matrices, inline validation, per-field reset, experiment
  kinematics editable), records browser with compare-overlay.
- **Aero coefficient registry** (`aero/registry.py`): named, cited,
  regime-tagged sets; `dickinson1999_robofly` and `flatplate_poststall_classical`
  (exact evaluation). Validity sections name the set in force.
- **Optional unsteady terms** (`aero/unsteady.py`): reduced-order wake
  capture and clap-and-fling (A-002 relaxed only when enabled; off by
  default; validity warning when active).
- **Control design loop** (`control/`): hover trim (bounded abdomen +
  wing-placement trim), linearization, LQR, closed-loop hover demo
  (`kind: closed_loop_reduced`).
- **System identification** (`analysis/sysid.py` + `fwmav-sim fit-sysid`):
  least-squares aero-coefficient fitting to calibration datasets with
  `source: fitted` write-back; synthetic recovery proof.
- **CI + docs:** GitHub Actions workflow (ruff + full pytest), this
  changelog, `docs/user_guide.md`.

## P1–P4 — Post-roadmap queue (2026-11-04)

- **P1:** wing structural mass in the dynamics (DR-006, A-010): planform
  thin-plate wing inertia folded into the base mass matrix; fixed-body
  inertial rig channels; AirPulse wing-mass variant (finding: the paper's
  38% assembly mass is inconsistent with its Table A1 inertia; the rig gap
  is NOT wing mass).
- **P2:** one-at-a-time parameter sensitivity (`fwmav-sim sensitivity`).
- **P3:** URDF import spike (DR-004 outcome; `fwmav-sim import-urdf`).
- **P4:** Level-3 abdomen aerodynamics (`aero/abdomen.py`; the M3 isolation
  flag is now live).
- GUI 3D playback (play/scrub/speed), experiment-open 3D preview, record
  vehicle-path fix, fullscreen-glitch fix, workflow tooltips.

## M0–M5 — Roadmap (2026-11)

- **M5:** PySide6 integrated workbench (inspector/editor/3D/timeseries/
  run/validity), headless-UI bit-identical equivalence, packaging.
- **M4:** dataset/alignment/filtering/metrics/compare pipeline; synthetic
  proof; AirPulse literature comparison (honest failure recorded).
- **M3:** articulated free body (DR-001 spatial dynamics, verified against
  the SymPy reference), abdomen control modes, mechanism isolation.
- **M2:** two-wing analytical model + abdomen; reduced longitudinal model;
  suspended-pendulum constraint model; RL-sandbox foil.
- **M1:** fixed-body blade-element aero (Dickinson fits, DR-003 AoA
  mapping), transform chain frozen (frames.md), convergence + regression.
- **M0:** modelling specification frozen (conventions, assumptions,
  decisions process).

## Perf pass — vectorized blade-element aero (2026-11-06)

- Blade-element kinematics + force evaluation vectorized over elements
  (batched arrays; the scalar path is kept and regression-pinned at 1e-12).
- Measured via `fwmav-sim benchmark`: m1_fixed_body_hover 12.54 s -> 1.00 s
  wall (12.5x; 0.33x realtime) on the dev machine.
- Unsteady terms stay on the scalar path (rare, off by default).
