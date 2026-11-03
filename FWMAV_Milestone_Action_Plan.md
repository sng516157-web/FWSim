# FWMAV Workbench — Milestone Action Plans for Kimi Code

**Status:** Ideation output, ready for execution
**Companion to:** `FWMAV_Simulator_Project_Brief.md` (the contract)
**Purpose:** Expand the brief's Milestones 0–5 into concrete, executable work packages: exact tech stack, repo layout, per-milestone task lists, test inventory, and reproduction commands. The brief remains the behavioural and scientific contract; this document is the build order.

**Update 2026-11-03:** Route A confirmed by Steven — portable pure-Python core with in-house spatial-vector dynamics (DR-001 stands); Linux-native/Drake deferred until ROS2/HITL or topology growth triggers a revisit. Per-stage executable prompts now live in `Milestones/` (`M0_…` through `M5_…`, see `Milestones/README.md`); those files are the stage-by-stage handoff to Kimi Code, this document remains the global reference.

**How Kimi Code should use this document:**

1. Execute milestones in order. Do not start M(N+1) until M(N)'s acceptance checks pass and its milestone report is written.
2. Where this document and the brief disagree, the brief wins — stop and flag the conflict instead of choosing silently.
3. Every pinned choice below carries a rationale and a revisit trigger. If a choice blocks progress, follow brief §14.11: document the evidence, propose a focused alternative, do not hide the problem.

---

## 1. Locked Technology Stack

All choices are **Windows-native, pip-installable, no conda required, no Linux-only runtime dependencies**. Verified 2026-11: Drake pip wheels are Linux/macOS-only; Pinocchio pip wheels are Linux-only (Windows requires conda-forge). Both are therefore excluded as runtime dependencies on the target machine.

### 1.1 Pinned core stack

| Layer | Choice | Version floor | Why | Rejected alternative |
|---|---|---|---|---|
| Language | Python | 3.12 | Brief §8.1; scientific ecosystem | MATLAB clone (explicit non-goal) |
| Packaging | `pyproject.toml` + hatchling, `pip install -e .[dev]` in `.venv` | — | Windows-native, zero toolchain friction | conda (not needed once Drake/Pinocchio excluded) |
| Arrays / numerics | NumPy | ≥ 1.26 | — | — |
| Integration / signal | SciPy (`integrate.solve_ivp`, `signal`) | ≥ 1.11 | Established integrators, per brief §5.7 | Hand-written RK (rebuilding generic infra, brief §3.6) |
| Config & schema | Pydantic v2 + PyYAML | ≥ 2.6 / ≥ 6.0 | Human-readable versioned definitions (§8.2) with enforced units/ranges; JSON Schema export for docs | Raw dicts; JSON-only |
| Symbolic cross-check (dev-only) | SymPy | ≥ 1.12 | Derive analytical reference EOM once, verify NumPy implementation against it | Derivation by hand only (unverifiable) |
| Multibody dynamics | **In-house spatial-vector algorithms (CRBA + RNEA) in NumPy**, restricted to floating base + small revolute chain | — | See §1.2 decision record | Drake (no Windows wheels), Pinocchio pip (Linux-only) |
| Testing | pytest + pytest-cov | ≥ 8.0 | — | unittest |
| Analysis plots | Matplotlib | ≥ 3.8 | M1–M4 static/phase plots | Plotly here (deferred to UI consideration) |
| 3D visualization | MeshCat (`meshcat` pip package, browser-based) | ≥ 0.3 | Pure-Python server, Windows-fine, browser rendering | Viser (evaluate in M5 spike only) |
| Experimental data import | pandas | ≥ 2.2 | CSV time-series, resampling | Hand-rolled CSV parsing |
| Lint / format | Ruff | ≥ 0.4 | One tool, fast | flake8+black combo |
| Version control | git; experiment records embed `git rev-parse HEAD` | — | Brief §3.5 traceability | — |

### 1.2 Decision record DR-001: multibody backend

**Decision:** Implement the articulated dynamics in-house using Featherstone spatial-vector algorithms (Composite Rigid Body Algorithm for M(q), Recursive Newton-Euler for C(q,v)v + G(q)), restricted to the actual MVP topology: one 6-DoF floating base + up to 2 revolute abdomen joints ≈ 8 generalized coordinates.

**Rationale:**

- The two "proven" candidates are not installable on the target Windows machine via pip (verified above).
- 8-DoF topology is far from a "general physics engine" (the thing brief §8.1 forbids building); CRBA+RNEA for a fixed chain is ~300 lines of auditable NumPy with direct access to every term — exactly what mechanism-isolation experiments (§6.6) need.
- No external engine means no black box between the equations and the validation tests.
- At 8 DoF, factorizing M(q) each step is cheap; ABA is unnecessary.

**Cross-validation (not a runtime dependency):** an optional dev-only script may install Pinocchio from conda-forge in a *separate* environment and diff M(q), C, G against the in-house implementation on random configurations. This is a test asset, never imported by the package.

**Revisit trigger:** topology grows past 2 internal joints, closed-loop linkages appear, or derivatives/optimization (trim, Floquet — deferred anyway) become the bottleneck. Re-spike then with conda Pinocchio or WSL2.

### 1.3 Decision record DR-002: UI shell (Milestone 5)

**Decision:** PySide6 desktop shell; time-series via pyqtgraph; 3D via MeshCat embedded in `QWebEngineView`; parameter editor auto-generated from the Pydantic schema.

**Rationale:** single Python process shares objects with the sim core, making headless/UI equivalence (M5 acceptance) trivial to test; Windows-native; no JS toolchain; MeshCat already chosen for M1–M3 3D checks.

**Rejected:** FastAPI + web frontend (duplicates state across a process boundary, heavier); a dashboard-first UI (explicitly forbidden by brief §9).

### 1.4 Explicitly not in the stack (guardrails from brief §4.2 / §14.12)

No CFD, no FSI, no flexible-wing predictive aero, no MATLAB interpreter, no block-diagram engine, no motor/gearbox model in the vertical slice, no photorealistic rendering. No dependency may be added without updating §1.1 with a rationale.

---

## 2. Repository Layout

```text
FlapSim/
├── pyproject.toml                 # name: fwmav-sim, console script: fwmav-sim
├── README.md
├── docs/
│   ├── conventions/
│   │   ├── frames.md              # M0 — every frame, axis, sign, rotation order
│   │   └── notation.md            # M0 — symbol table used in code docstrings
│   ├── assumptions.md             # assumption register, IDs A-001…, cited in code
│   ├── test_plan.md               # maps brief §11 → test files
│   ├── decisions/                 # DR-001, DR-002, future decision records
│   └── milestone_reports/         # one per milestone, per brief §14.10
├── derivations/                   # SymPy scripts → analytical references (dev-only)
│   └── reduced_longitudinal.py
├── reference_vehicles/
│   └── bio_abdomen_fwmav.yaml     # §12 vehicle; unavailable params marked "estimated/pending"
├── experiments/                   # runnable YAML experiment definitions
├── src/fwmav_sim/
│   ├── core/                      # frames, transforms, quaternions, time series container
│   ├── vehicle/                   # Pydantic definition schema, bodies, joints, mass props
│   ├── kinematics/                # prescribed wing kinematics, blade-element states
│   ├── aero/                      # AerodynamicModel interface + quasi-steady blade element
│   ├── dynamics/                  # reduced analytical model; spatial-vector multibody
│   ├── actuation/                 # prescribed → profile-limited → torque-driven stages
│   ├── control/                   # controller interface (PID/LQR later)
│   ├── environment/               # still air, uniform wind
│   ├── constraints/               # fixed-body, suspended-string, free-body
│   ├── simulation/                # runner, solver config, experiment orchestration
│   ├── experiment/                # ExperimentRecord: provenance, save/load
│   ├── datasets/                  # M4 — experimental import, alignment, filtering
│   ├── analysis/                  # averaging, metrics, residuals, convergence studies
│   ├── viz/                       # matplotlib plots; meshcat 3D scenes
│   ├── ui/                        # M5 — PySide6 shell only
│   └── cli.py                     # fwmav-sim run / validate / report
└── tests/
    ├── unit/                      # §11.1
    ├── analytical/                # §11.2
    ├── numerical/                 # §11.3
    └── validation/                # §11.4 — may contain pending/skipped-with-reason tests
```

**Naming rule:** every physical symbol in code matches `docs/conventions/notation.md`. A docstring that introduces a symbol not in the table is a review blocker.

---

## 3. Cross-Cutting Conventions (apply to every milestone)

1. **Units:** SI internally, always. The schema records the documented unit per parameter; experimental import (M4) converts at the boundary with the conversion recorded in the dataset metadata.
2. **Provenance:** every physical parameter carries `{value, unit, source ∈ {measured, literature, estimated, fitted}, uncertainty_or_range, calibration_dataset, valid_regime}` per brief §5.8. Schema-enforced, not documentation-enforced.
3. **Experiment record:** every run saves config YAML + git hash + solver settings + RNG seed + output schema version to `outputs/<timestamp>_<name>/`. Headless rerun must reproduce it (brief §8.2, §9 step 11).
4. **Honesty hooks:** any code path that produces output under an unsupported regime (e.g., large advance ratio, wing deformation requested) emits a visible warning into the run record and the UI validity panel (brief §3.5, §14.6).
5. **Failure behaviour:** non-finite state, excessive motion, or solver failure → run stops, record marked `FAILED` with the failing state dumped. Never plausible-looking partial output (M3 acceptance).
6. **Windows/Git Bash note for Kimi Code:** use `pwd -W` for Windows paths; quote paths with spaces.

---

## 4. Milestone Action Plans

Each plan: goal → tasks (in order) → test inventory → acceptance commands → report requirements. Estimated effort assumes one focused implementer.

---

### Milestone 0 — Modelling specification and test plan

**Goal:** a second developer can determine every frame, state and force direction without reading code (brief §10 M0). Est. 3–5 days.

**Tasks**

1. **Scaffold.** `pyproject.toml` (hatchling, Python 3.12, deps from §1.1), ruff config, pytest config, `src/` layout, `.venv` setup. Verify: `pip install -e .[dev]`, `pytest --collect-only` runs clean.
2. **`docs/conventions/frames.md`.** Mandate this exact convention (no ambiguity left):
   - **World frame W:** NED (x-north, y-east, z-down); gravity vector `g = [0, 0, +9.81] m/s²`.
   - **Body frame B:** x-forward, y-right, z-down; coincides with W at zero attitude.
   - **Attitude:** ZYX Euler (yaw ψ, pitch θ, roll φ_r); quaternion scalar-first `(w, x, y, z)`, Hamilton; stored normalized, renormalized every step.
   - **Stroke-plane frame SP:** B rotated by stroke-plane angle β about B-y.
   - **Wing-root frame WR:** span axis along +y (right wing), chord along x, per Sane & Dickinson convention; left wing handled by explicit reflection operator with documented sign rules (§5.2 symmetry requirement).
   - **Wing angles:** flapping (stroke position) φ, pitching (rotation) θ_w, deviation ψ_w, with sign definitions and zero references drawn in the doc.
   - **Blade-element frame BE:** per element, origin at element centre.
   - Every transform chain written as `W ← B ← SP ← WR ← BE` with rotation-matrix formulas.
3. **`docs/conventions/notation.md`:** symbol table (φ, θ_w, β, α_abd, q, v, τ, …) matching future code names.
4. **Reduced-model spec section:** state vector for the longitudinal thorax–abdomen model `[x, z, θ, α_abd]`, full EOM in symbolic form, small-angle linearization, and the two-mass point-mass assumptions — written *before* M2 code exists.
5. **Parameter schema.** Pydantic models: `VehicleDefinition`, `BodySpec`, `JointSpec`, `WingSpec`, `BladeElementSpec`, `AeroModelSpec`, `ActuatorSpec`, `SolverSpec`, `EnvironmentSpec`, `Provenance` (per §3 item 2). Include JSON Schema export to `docs/schema/`.
6. **`docs/assumptions.md`:** register with IDs; seed with: rigid wings (A-001), no wake capture/clap-and-fling (A-002), quasi-steady regime (A-003), prescribed wing kinematics (A-004), no wing deformation (A-005), uniform inflow not modelled beyond relative velocity (A-006).
7. **`docs/test_plan.md`:** table mapping every bullet of brief §11.1–11.4 to a concrete test file and test function name (stubs allowed).
8. **`docs/milestone_reports/template.md`:** implements brief §14.10 (capabilities / tests run + actual results / known failures / deferred / reproduction commands).

**Tests (stubs counted, real where possible)**

- `tests/unit/test_schema.py` — schema rejects missing units, unknown provenance source, out-of-range values.
- `tests/unit/test_frames_stub.py` — placeholder asserting the frame registry lists all six frames.

**Acceptance commands**

```bash
pip install -e .[dev]
pytest tests/unit -q
python -c "from fwmav_sim.vehicle.schema import VehicleDefinition; print(VehicleDefinition.model_json_schema()[\"title\"])"
```

**Report:** `docs/milestone_reports/M0.md`, including the "second developer" checklist from brief §10 M0.

---

### Milestone 1 — Fixed-body aerodynamic vertical slice

**Goal:** one rigid wing, prescribed kinematics, blade elements, separable quasi-steady force terms, convergence evidence (brief §10 M1). Est. 1.5–2.5 weeks.

**Tasks**

1. **`core/transforms.py`.** Rotation matrices, quaternion ops, angular-velocity-to-quaternion-rate, frame-chain composition. Property-style tests: matrix↔quaternion round trip; `R(a)R(b) = R(ab)`; numerical vs analytical derivative of `R(t)` at `1e-6` agreement.
2. **`kinematics/prescribed.py`.** Sinusoidal flapping `φ(t) = φ_amp·cos(2πft)`, wing pitching `θ_w(t)` with configurable amplitude/phase/offset (including square/triangle-ish profiles via Fourier terms), optional deviation. Stroke-plane angle β. Left/right mirroring through the documented reflection operator.
3. **`kinematics/blade_element.py`.** Discretize span into N elements (equal-area option); compute per-element position, velocity, acceleration, local chord/area, AoA, Re — in the BE frame, consuming only the transform chain (brief §5.2: all aero models eat the same kinematic state).
4. **`aero/interface.py`.** `class AerodynamicModel: def element_forces(state: ElementState, env: Environment) -> ElementForces` where `ElementForces` returns **translational, rotational, added-mass** as separate vectors plus resultant (brief §5.1).
5. **`aero/quasi_steady.py`.** Extended quasi-steady model with **cited, configurable coefficients**:
   - Translational lift/drag: Dickinson et al. 1999 fit (`CL = 0.225 + 1.58·sin(2.13α − 7.20°)`, `CD = 1.92 − 1.55·cos(2.04α − 9.82°)`) as the default; coefficients live in YAML with provenance=literature.
   - Rotational circulation: Sane & Dickinson 2002 form.
   - Added mass: flat-plate per-unit-span `ρ·π·c²/4` normal-acceleration term (Sedov/Ellington lineage).
   - Every equation's docstring cites source + assumption IDs. No equations from memory (brief §16).
6. **`environment/still_air.py`, `environment/uniform_wind.py`.** Relative airflow = element velocity − air velocity (brief §5.4).
7. **`simulation/fixed_body.py`.** Prescribed kinematics ⇒ direct time-stepping (no ODE solve needed); configurable steps-per-wingbeat; outputs a `TimeSeries` (instantaneous per-term forces, resultant, power estimate where modelled).
8. **`analysis/averaging.py`.** Wingbeat averaging from the same time-resolved arrays (brief §3.3); phase-aligned cycle segmentation.
9. **`viz/force_plots.py`.** Matplotlib: Fx/Fz vs phase, per-term overlay, cycle-averaged bars.
10. **MeshCat spike + minimal scene** (`viz/meshcat_scene.py`): wing as discretized surface, BE frames, per-element velocity and force arrows, CoM marker. Deliverable includes install/runtime check on Windows; if MeshCat fails, decision record + fallback (matplotlib 3D) — do not silently drop 3D.
11. **Reference vehicle draft** `reference_vehicles/bio_abdomen_fwmav.yaml`: geometry placeholders clearly `source: estimated`, pending Steven's parameter table (see §6 questions).

**Tests**

- Unit (§11.1): transforms round trips; kinematic derivatives vs finite difference; blade-element velocity/AoA against hand-computed element; each aero term against hand-computed values at α = 0°, 45°, 90°.
- Analytical (§11.2): stationary wing + still air ⇒ zero force; uniform translation ⇒ steady drag with correct sign; mirrored wings ⇒ mirrored forces; reversed φ_amp sign ⇒ documented sign change.
- Numerical (§11.3): blade-count sweep N ∈ {4, 8, 16, 32, 64} and steps/wingbeat ∈ {50, 100, 200, 400, 800, 1600}; produce convergence table (L2 difference of cycle-averaged forces vs finest case) saved into the run record. Script: `python -m fwmav_sim.analysis.convergence experiments/m1_convergence.yaml`.

**Acceptance commands**

```bash
pytest tests/unit tests/analytical tests/numerical -q
fwmav-sim run experiments/m1_fixed_body_hover.yaml
fwmav-sim report outputs/<run_dir>   # regenerates plots from stored data, no hard-coded curves
```

**Report:** M1 report includes the convergence table verbatim, per-term force plots, and the explicit statement that wake capture/clap-and-fling/deformation are excluded (brief §5.1, §5.3).

---

### Milestone 2 — Two-wing vehicle and analytical abdomen model

**Goal:** two wings + longitudinal thorax–abdomen analytical model with verified CoM/inertia and pendulum behaviour (brief §10 M2). Est. 1.5–2 weeks.

**Tasks**

1. **`vehicle/definition.py`.** Load full vehicle YAML: thorax, abdomen (concentrated mass + optional distributed), two wings, joint definitions (abdomen pitch; roll added to schema now, exercised in M3).
2. **`dynamics/mass_properties.py`.** `com(q)` and `inertia_about_com(q)` implementing brief §6.2 exactly (parallel-axis form). Must derive from geometry/config, never from a viz marker.
3. **`derivations/reduced_longitudinal.py` (SymPy).** Derive the two-mass longitudinal EOM (states `[x, z, θ, α_abd]`) via Euler–Lagrange; export symbolic expressions; generate a numeric reference module. This script is the *independent analytical reference* required by brief §15.
4. **`dynamics/reduced_longitudinal.py` (NumPy).** Hand-implemented EOM; test asserts agreement with the SymPy-generated reference at 100 random states to `< 1e-10` (relative).
5. **`constraints/suspended.py`.** Suspended-string pivot as an explicit constraint model (pendulum dynamics, brief §6.3) — separate from free-body, never conflated.
6. **`actuation/prescribed_abdomen.py`.** Stage 1 of brief §6.4: prescribed abdomen angle α_abd(t); record commanded vs actual.
7. **Cycle-averaged wing-force injection:** reduced model consumes averaged forces derived from an M1 time-resolved run (brief §3.3), loaded as data — not recomputed by an unrelated formula.
8. **Passive and actuated pendulum cases** as runnable experiments: `experiments/m2_passive_pendulum.yaml`, `m2_actuated_pendulum.yaml`.

**Tests**

- Unit (§11.1): CoM/inertia vs hand calculation on 3 simple configurations (symmetric, abdomen-deflected, point-mass limit); joint-limit clamping.
- Analytical (§11.2): free-body gravity moment about instantaneous CoM ≡ 0 (machine precision); passive pendulum small-oscillation frequency vs analytical `ω = sqrt(m g d / I)` to `< 1e-6` relative; linearized model vs numerical Jacobian; symmetric flapping ⇒ zero net roll/yaw moment.
- Numerical (§11.3): frictionless pendulum energy drift `< 1e-8` per period (DOP853, rtol 1e-9); timestep refinement table.

**Acceptance commands**

```bash
python derivations/reduced_longitudinal.py   # regenerates reference, asserts match
pytest tests/unit tests/analytical tests/numerical -q
fwmav-sim run experiments/m2_passive_pendulum.yaml
fwmav-sim run experiments/m2_actuated_pendulum.yaml
```

**Report must state** (brief §10 M2 acceptance): pendulum agreement is **not** free-flight validation; Leng et al. numerical reproduction remains `PENDING` until original parameter values/time series are supplied (brief §12).

---

### Milestone 3 — Nonlinear articulated free-body simulation

**Goal:** 6-DoF thorax + abdomen pitch/roll, coupled aero-multibody dynamics, mechanism isolation (brief §10 M3). Est. 3–4 weeks.

**Tasks**

1. **M3.0 backend confirmation spike (timeboxed 2 days).** Execute DR-001: implement minimal CRBA on the 8-DoF chain; (optional, if conda available) diff against conda-forge Pinocchio on random configurations. Write `docs/decisions/DR-001-outcome.md`. If the in-house route shows a correctness or performance blocker, escalate per brief §14.11 with the conda/WSL2 Pinocchio fallback — do not integrate two engines at once.
2. **`dynamics/spatial.py`.** Spatial vectors, motion/force transforms, per-joint S matrices; **CRBA** (M(q)) and **RNEA** (bias C(q,v)v + G(q) and inverse dynamics for reaction torques); solve `M q̈ = τ + JᵀF − bias` by direct factorization (8×8 — no ABA needed). Reaction torques at abdomen joints returned explicitly (brief §6.1 item 3).
3. **Floating-base integration on the manifold:** position + quaternion state; `q̇` from body rates; renormalize quaternion each accepted step; norm check test.
4. **Aero coupling.** Element forces from M1's model mapped through element Jacobians `J_a(q)ᵀ F_aero` (brief §5.4); relative airflow includes 6-DoF body motion.
5. **`actuation/profile_limited.py`.** Stage 2 of brief §6.4: rate/acceleration-limited abdomen profiles; **`actuation/torque_driven.py`:** stage 3, torque-driven joint with damping and limits. Commanded/actual position, velocity, torque, saturation all recorded.
6. **Mechanism-isolation switches** (brief §6.6) as first-class config flags on the experiment schema: `mass_shift_enabled`, `inertial_coupling_enabled`, `abdomen_aero_enabled` (abdomen aero itself is Level-3 — the flag exists, the model is a documented stub), plus the combined case. One YAML per case under `experiments/m3_isolation/`.
7. **`simulation/runner.py`.** `solve_ivp` (DOP853 default; configurable rtol/atol; `max_step = wingbeat_period / steps_per_wingbeat`); event hooks; NaN/excessive-motion guard ⇒ hard stop with `FAILED` record (§3 item 5).
8. **`experiment/record.py`.** Full save/load: config, provenance, solver settings, git hash, RNG seed, outputs as `.npz` + metadata JSON. `fwmav-sim rerun <record_dir>` reproduces.

**Tests**

- Analytical: planar restriction of the full model reproduces the M2 reduced model's trajectories to `< 1e-6`; free-body gravity moment about CoM ≡ 0; aero-off free flight ⇒ linear momentum follows gravity exactly and angular momentum conserved to solver tolerance.
- Numerical: quaternion norm `|1 − ‖q‖| < 1e-12` post-step; timestep refinement does not change declared metrics within tolerance (table in report); save/reload + headless rerun ⇒ outputs identical for fixed-step, within solver tolerance for adaptive (exact rule documented).
- Failure-injection: NaN-forcing config ⇒ run stops visibly, record `FAILED` (asserted in test).

**Acceptance commands**

```bash
pytest tests/ -q
fwmav-sim run experiments/m3_free_body_hover.yaml
fwmav-sim run experiments/m3_isolation/geometry_only.yaml
fwmav-sim rerun outputs/<previous_run>
```

**Report:** isolation-case comparison table (more informative than one fitted curve — brief §6.6); lever-arm and joint-torus time series exposed; solver-failure behaviour demonstrated.

---

### Milestone 4 — Validation and experiment comparison

**Goal:** native import/align/compare workflow with honest labels (brief §10 M4). Est. 1.5–2 weeks. **Blocked-input risk:** Leng et al. time-series data — see §6.

**Tasks**

1. **`datasets/importer.py`.** pandas-based CSV import; per-column unit declaration; quantity mapping to simulated equivalents (force at load cell ≠ aerodynamic force — inertial content handled explicitly per brief §5.9); dataset labelled `calibration` or `validation` at import, schema-enforced.
2. **`datasets/alignment.py`.** Wingbeat-phase synchronization via cross-correlation on the dominant flapping harmonic; resampling onto common phase grid; time offset + drift reported.
3. **Filtering:** `scipy.signal.filtfilt` with documented filter spec; the same filter applied to simulated and experimental series where applicable; full filter metadata stored (brief §5.9 item 3).
4. **`analysis/metrics.py`.** RMSE, NRMSE, MAE, correlation, phase lag, Welch spectra comparison; residual time series.
5. **`viz/comparison.py`.** Overlay plots, residual plots, spectra; figures distinguish aerodynamic / inertial / measured-resultant where relevant (M4 acceptance).
6. **Synthetic sanity dataset:** generate a known-truth synthetic "experiment" from the simulator itself (clearly labelled synthetic) to prove the pipeline end-to-end without claiming validation.
7. **Validation report generator** (`fwmav-sim compare <exp_record> <dataset>` → report dir): metrics table, residuals, dataset labels, dataset provenance. No "validated" badge without a real dataset + defined metric (M4 acceptance); pending status otherwise.

**Tests**

- Unit: importer unit conversion; alignment recovers known synthetic phase offset; metrics against hand-computed values.
- Numerical: synthetic-dataset round trip ⇒ near-zero residuals (up to added known noise).
- Validation (§11.4): Leng et al. comparison present as `pytest.mark.skip(reason="awaiting original data")` until data arrive.

**Acceptance commands**

```bash
pytest tests/ -q
fwmav-sim compare outputs/<m3_run> datasets/synthetic_hover.csv
```

---

### Milestone 5 — Integrated workbench

**Goal:** a new user runs the full §9 workflow without editing source; UI ≡ headless numerically (brief §10 M5). Est. 3–4 weeks. Execute DR-002.

**Tasks**

1. **PySide6 shell** (`ui/`): vehicle/tree inspector; parameter editor generated from the Pydantic schema (units + valid ranges shown, provenance editable); run-status panel; assumption/validity panel fed by the honesty hooks (§3 item 4).
2. **pyqtgraph** time-series views reusing `analysis/` products; **MeshCat** scene in `QWebEngineView` with CoM / aerodynamic-centre / lever-arm overlays (brief §9 UI priorities).
3. **Experiment save/load** in the UI on top of `experiment/record.py`; headless equivalence test: same YAML via UI and via CLI ⇒ outputs equal within documented tolerance.
4. **Workflow wiring** for brief §9 steps 1–11 exactly, including the step-11 headless repeatability check as an automated test.
5. **Packaging:** `pip install fwmav-sim` from a clean venv ⇒ workflow runs on the reference vehicle; README quickstart.

**Tests:** headless/UI equivalence; every demo plot generated from current model output (test regenerates a UI plot from a fresh run and compares against the displayed buffer); full §9 workflow as an integration test.

**Acceptance:** brief §10 M5 checklist verbatim in the milestone report.

---

## 5. Dependency Add/Change Protocol

Any new dependency requires: (1) entry in §1.1 with rationale and rejected alternative; (2) Windows pip-installability verified; (3) a note in the current milestone report. Adding a heavy backend (Drake/Pinocchio/etc.) requires a decision record and Steven's sign-off.

---

## 6. Open Questions for Steven

1. **Reference-vehicle parameters and data (blocks full M2/M4 numerical reproduction):** the brief names Leng, Ng & Tang (IMAV 2024) as the reference vehicle and says to request original parameter values and time series before numerical reproduction (§12). Can you supply the paper's parameter table (masses, geometry, wing kinematics, aerodynamic coefficients used) and the suspended-string transient time series? Without them, M2/M4 ship with the test structure complete and data-dependent validation explicitly `PENDING` — which the brief permits, but the real comparison is the interesting part.
2. **UI shell:** DR-002 picks PySide6 desktop. If you'd rather have a browser-based workbench (e.g., for sharing with collaborators), say so before M5 starts — it changes M5 only, not M0–M4.
3. **Cross-check environment:** is a conda installation available on your machine for the optional dev-only Pinocchio cross-check in M3.0? If not, the spike proceeds with the in-house implementation alone plus its analytical regression tests.
4. **Repo hosting/license:** private or public? License choice affects whether Dickinson-fit coefficients and any paper-derived data can be committed directly.

---

## 7. Risk Register (top 5)

| Risk | Impact | Mitigation in plan |
|---|---|---|
| Quasi-steady coefficients wrong for the vehicle's Re/kinematics | Silent force error | Provenance-tracked coefficients; M4 residual workflow; assumption panel; no "validated" claim without data |
| In-house multibody bug | Wrong coupled dynamics | SymPy-derived analytical reference (M2) + planar-limit recovery test (M3) + momentum/energy conservation tests + optional Pinocchio diff |
| Transform/sign-convention error | Plausible-but-wrong forces (brief §5.2) | M0 convention doc frozen first; finite-difference and symmetry tests in M1 before any coupling |
| Reference data never arrives | M4 stays structural | Explicit `PENDING` labels; synthetic-dataset pipeline proof; brief-compliant honest reporting |
| Scope creep toward UI/CFD/block editors | P0 dynamics never finish | Guardrails §1.4; milestone-gate rule (brief §14.1, §14.12) restated at the top of this document |

---

## 8. Reference Projects

External code reviewed for reuse, with an explicit verdict per project. Nothing enters the runtime path without appearing in §1.1.

### 8.1 `sng516157-web/2D-RL-Control-for-a-flapping-wing-drone` (Steven's RL sandbox, reviewed 2026-11-03)

2D flapping-wing MAV + REINFORCE controller (PyTorch/pygame, MIT license). Local reference copies: `reference_rl_README.md`, `reference_rl_flapping.py`.

**Not reusable — the plant.** Thrust linear in frequency (`T = k·f`; the repo's own README flags `T ∝ f²` as the needed fix), Euler integration, and a sinusoidal-plus-Gaussian "flutter" torque standing in for physics would violate brief §3.3 (averaged forces must derive from time-resolved runs), §5.7 (established integrators), and §8.2/§14.5 (no hard-coded plausible behaviour as dynamics). No line of its `flapper_step` migrates into FlapSim. Retained value here: a documented anti-pattern for the assumption/validity panel — what a cycle-averaged, no-blade-element toy model looks like.

**Reusable — four items, all deferred to the P1 "controller interface and strategy comparison" work (after M3/M4):**

1. **Controller-interface template.** Its gym-style environment (error-based observation, normalized actions in [-1,1]², gain+clamp mapping to physical control variables) is the pattern for FlapSim's `Controller` interface (brief §8). Its HUD metrics (tracking error, action effort, frequency² effort) map almost one-to-one onto brief §7's common metrics.
2. **RL as a compared control strategy.** Once the M3 articulated plant exists, train PPO (not REINFORCE — known high variance) against it, especially for abdomen/CoG-shift control where PID gain structure is non-obvious. Target library when the time comes: stable-baselines3 — to be pinned via the §5 dependency protocol then, not now.
3. **M2 validation foil.** With noise disabled and parameters mapped, the M2 reduced analytical model (brief §6.5 Level 1) should reproduce the sandbox's qualitative behaviour — hover equilibrium `f_hover = mg/(2k)`, differential-frequency attitude response. Cheap integration test, worth wiring when M2 lands.
4. **Benchmark-task design.** Its moveable setpoint and automatic oval routine are better controller-comparison tasks than step responses alone; adopt into the P1 comparison harness.

**Status:** recorded, deferred. Do not pull into M0–M5 scope.
