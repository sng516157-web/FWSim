# Milestone 0 — Modelling Specification and Test Plan

> **Stage prompt for Kimi Code.** Read these two documents first — they are the contract:
> - `../FWMAV_Simulator_Project_Brief.md` (behavioural and scientific contract)
> - `../FWMAV_Milestone_Action_Plan.md` (build order, locked stack §1, conventions §3)
>
> Where this file and the brief disagree, the brief wins. Stop and flag the conflict; never choose silently (brief §14.7).

**Goal:** a second developer can determine every frame, state and force direction without reading implementation code (brief §10 M0).
**Estimated effort:** 3–5 days.
**Route A applies:** pure-Python portable core, Windows-native, no Drake/Pinocchio/conda dependencies anywhere in this project.

---

## 0.1 Ground rules for this stage

- This milestone produces **documentation, schema and scaffolding only**. No dynamics, no aerodynamics, no UI code beyond package skeletons and test stubs.
- Follow brief §14: write tests alongside each subsystem (here: schema tests); never fabricate parameters; label every placeholder.
- All units SI internally. Angle signs, axis directions and rotation order are defined in this milestone and are **frozen** afterwards — later milestones must not redefine them.
- Windows/Git Bash environment: use `pwd -W` for Windows paths; quote paths containing spaces.

## 0.2 Locked stack for this stage

Python 3.12 · pip + `.venv` · hatchling `pyproject.toml` · NumPy ≥ 1.26 · Pydantic ≥ 2.6 · PyYAML ≥ 6.0 · pytest ≥ 8.0 + pytest-cov · Ruff ≥ 0.4. SymPy ≥ 1.12 goes in the `dev` extra now (used from M2 onward).

---

## 0.3 Tasks (in order)

### Task 1 — Repository scaffold

Create the layout from action plan §2 exactly:

```text
pyproject.toml  README.md  docs/  derivations/  reference_vehicles/
experiments/  src/fwmav_sim/  tests/{unit,analytical,numerical,validation}/
```

- `pyproject.toml`: project name `fwmav-sim`, Python 3.12, runtime deps (numpy, scipy, pydantic, pyyaml, matplotlib, meshcat, pandas), `dev` extra (pytest, pytest-cov, ruff, sympy), console script `fwmav-sim = fwmav_sim.cli:main`.
- Empty subpackages with `__init__.py` for: `core vehicle kinematics aero dynamics actuation control environment constraints simulation experiment datasets analysis viz ui`.
- Ruff config (line length 100, select E/F/I/UP), pytest config (`testpaths = tests`, strict markers).
- Verify: `pip install -e .[dev]` succeeds; `pytest --collect-only` exits 0.

### Task 2 — `docs/conventions/frames.md` (the core deliverable)

Mandate **exactly** this convention — no alternatives left open:

- **World frame W:** NED — x north, y east, z **down**; gravity vector `g = [0, 0, +9.81] m/s²`.
- **Body frame B:** x forward, y right, z down; coincides with W at zero attitude.
- **Attitude:** ZYX Euler (yaw ψ → pitch θ → roll φ_r), intrinsic; quaternion scalar-first `(w, x, y, z)`, Hamilton convention, stored normalized.
- **Stroke-plane frame SP:** B rotated by stroke-plane angle β about the B y-axis; document the positive-β direction with a diagram.
- **Wing-root frame WR:** span axis along +y for the right wing, chord along x (Sane & Dickinson convention). Left wing obtained by an explicit reflection operator `S = diag(1, -1, 1)` applied at the documented point in the transform chain, with the sign rules for angles and velocities written out.
- **Blade-element frame BE:** origin at element centre, axes aligned with WR.
- **Wing angles:** flapping (stroke position) φ, wing pitching θ_w, deviation ψ_w. Give zero references, positive directions, and the time-domain definitions for prescribed sinusoidal kinematics.
- **Transform chains:** write `W ← B ← SP ← WR ← BE` as explicit 3×3 rotation-matrix formulas, plus the position/velocity/acceleration transport equations.
- ASCII or TikZ diagrams for every frame. Include a "common mistakes" section (e.g., gravity sign, quaternion order, left-wing mirroring).

### Task 3 — `docs/conventions/notation.md`

Symbol table binding every physical symbol to its future Python name: `phi` (flapping), `theta_w` (wing pitch), `beta` (stroke plane), `alpha_abd` (abdomen pitch), `q, v, tau`, `R_W_B`, `F_trans, F_rot, F_am`, etc. Rule to state in the doc: code introducing a symbol not in this table is a review blocker.

### Task 4 — Reduced-model specification section

In `frames.md` or a sibling `docs/conventions/reduced_model_spec.md`, specify **before any M2 code exists**:

- State vector of the longitudinal thorax–abdomen model: `x = [x, z, θ, α_abd, ẋ, ż, θ̇, α̇_abd]`.
- Point-mass assumptions for thorax and abdomen, joint location, CoM definition.
- Full nonlinear EOM in symbolic form (Euler–Lagrange), plus the small-angle linearization and its state-space matrices.
- Definitions: pitch zero reference, abdomen angle zero reference, thrust direction convention, CoM reference point — brief §10 M0 forbids ambiguity on any of these.

### Task 5 — Parameter schema (`src/fwmav_sim/vehicle/schema.py`)

Pydantic v2 models, every physical parameter carrying provenance (brief §5.8):

```python
class Provenance(BaseModel):
    source: Literal["measured", "literature", "estimated", "fitted"]
    uncertainty_or_range: str | None = None
    calibration_dataset: str | None = None
    valid_regime: str | None = None
    reference: str | None = None        # citation for literature values

class Parameter(BaseModel):
    value: float
    unit: str                            # SI; required, non-empty
    provenance: Provenance
```

Then: `BodySpec` (mass, CoM offset, inertia tensor), `JointSpec` (type, axis, limits, damping), `WingSpec` (span, root chord, tip chord, planform, hinge location), `BladeElementSpec`, `AeroModelSpec` (model name + coefficient set), `ActuatorSpec`, `SolverSpec` (integrator, rtol, atol, steps_per_wingbeat, max_time), `EnvironmentSpec` (air density, wind vector), `VehicleDefinition` (bodies, joints, wings, aero selection). Export JSON Schema to `docs/schema/` via a small script.

### Task 6 — `docs/assumptions.md`

Assumption register with stable IDs, seeded with: A-001 rigid wings · A-002 no wake capture / clap-and-fling · A-003 quasi-steady regime · A-004 prescribed wing kinematics · A-005 no wing deformation · A-006 no inflow model beyond relative velocity · A-007 cycle-averaged forces in reduced model only. Rule: code relying on an assumption cites its ID in the docstring.

### Task 7 — `docs/test_plan.md`

Table mapping **every** bullet of brief §11.1–11.4 to a concrete test file + test function name (stubs acceptable where the code lands in later milestones). Example rows:

| Brief § | Test | File |
|---|---|---|
| 11.1 rotation transforms | `test_quat_matrix_roundtrip` | `tests/unit/test_transforms.py` |
| 11.2 gravity moment about CoM ≡ 0 | `test_gravity_moment_about_com_zero` | `tests/analytical/test_mass_properties.py` |
| 11.3 timestep refinement | `test_timestep_refinement` | `tests/numerical/test_convergence.py` |

### Task 8 — `docs/milestone_reports/template.md`

Template implementing brief §14.10: implemented capabilities / tests run **with actual numeric results** / known failures and limitations / deferred items / exact reproduction commands / software revision (git hash).

---

## 0.4 Tests for this milestone

- `tests/unit/test_schema.py` — schema rejects: missing unit, unknown provenance source, negative mass, malformed inertia tensor (non-symmetric or non-positive-definite where checkable).
- `tests/unit/test_frames_registry.py` — a `core/frames.py` registry lists all six frames with parent/child relationships matching `frames.md`.
- `pytest --collect-only` must list every test stub named in `test_plan.md`.

## 0.5 Acceptance commands

```bash
pip install -e .[dev]
ruff check src tests
pytest tests/unit -q
pytest --collect-only -q
python -c "from fwmav_sim.vehicle.schema import VehicleDefinition; print(VehicleDefinition.model_json_schema()['title'])"
```

## 0.6 Milestone report

Write `docs/milestone_reports/M0.md` from the template, including the brief §10 M0 self-check: could a second developer determine every frame, state and force direction from these docs alone? List anything still ambiguous explicitly.

## 0.7 Explicit non-goals for this stage

No equations of motion code, no aerodynamic code, no integrators, no visualization, no UI. Schema + docs + stubs only.
