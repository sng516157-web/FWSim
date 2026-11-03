# Milestone 3 — Nonlinear Articulated Free-Body Simulation

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — especially §5.4, §5.6, §5.7, §6 (all), §10 M3, §11
> - `../FWMAV_Milestone_Action_Plan.md` — stack §1, **decision record DR-001 §1.2** (multibody backend), conventions §3
> - `docs/conventions/` and M2's `dynamics/reduced_longitudinal.py` + SymPy reference — your regression anchor
>
> Conflicts: the brief wins; stop and flag. Route A: **in-house spatial-vector dynamics in NumPy (DR-001). No Drake, no Pinocchio, no conda in the runtime path.**

**Goal:** 6-DoF thorax + abdomen pitch and roll, coupled aero-multibody dynamics, prescribed and motion-limited abdomen inputs, mechanism-isolation runs (brief §10 M3).
**Estimated effort:** 3–4 weeks.
**Prerequisite:** M2 accepted (verified reduced model and mass properties).

---

## 3.1 Ground rules for this stage

- Fixed-body (M1) and constrained (M2) modes remain available as **separate test configurations** — do not treat them as the same plant (brief §5.4).
- The simulator must expose force-to-CoM lever arms and internal joint reaction torques (brief §10 M3) — these are outputs, not internals.
- Solver failure, non-finite state, or excessive motion ⇒ the run stops visibly and the record is marked `FAILED`. Never emit plausible-looking partial output (brief §10 M3 acceptance).
- Abdomen aerodynamics itself is Level 3 (brief §6.5): the isolation **flag** exists this milestone; the aero model is a documented stub that raises if enabled without a model.

## 3.2 Locked stack for this stage

Unchanged: NumPy/SciPy/Pydantic/YAML/pytest/Matplotlib/MeshCat. Integrator: `scipy.integrate.solve_ivp`, default DOP853, configurable rtol/atol, `max_step = wingbeat_period / steps_per_wingbeat`.

---

## 3.3 Tasks (in order)

### Task 1 — M3.0 backend confirmation spike (timeboxed 2 days)

Execute DR-001: implement minimal CRBA on the 8-DoF chain (floating base + 2 revolute joints) and validate M(q) against the M2 SymPy-derived mass matrix in the planar restriction. If conda happens to be available, a dev-only script may diff M/C/G against conda-forge Pinocchio on 100 random configurations — **test asset only, never imported by the package**. Write `docs/decisions/DR-001-outcome.md` with actual numbers. If a correctness or performance blocker appears, escalate per brief §14.11 (propose WSL2/conda Pinocchio fallback) — do not integrate two engines at once (brief §8.1).

### Task 2 — `dynamics/spatial.py` (the core)

Featherstone spatial-vector algebra, restricted to floating base + small revolute chain:

```python
class ArticulatedModel:          # built from VehicleDefinition
    def crba(self, q) -> np.ndarray: ...                       # M(q), (n,n)
    def bias(self, q, v) -> np.ndarray: ...                    # C(q,v)v + G(q) via RNEA with a=0
    def forward_dynamics(self, q, v, tau, f_ext) -> np.ndarray: ...  # solve M q̈ = τ + JᵀF − bias
    def reaction_torques(self, q, v, a, f_ext) -> np.ndarray: ...    # internal joint torques
    def jacobian_com(self, q, body_id) -> np.ndarray: ...      # for aero force mapping & lever arms
```

Floating base: position + quaternion state; `q̇` from body rates; quaternion renormalized every accepted integrator step. At n = 8, direct factorization of M — **no ABA needed**.

### Task 3 — Manifold-aware integration (`simulation/integrator.py`)

State packing/unpacking for `(p, quat, q_joints, v, ω, ω_joints)`; post-step quaternion normalization; norm guard `|1 − ‖q‖| < 1e-12` enforced by test.

### Task 4 — Aero coupling

Element forces from M1's quasi-steady model mapped through element Jacobians: generalized force contribution `J_a(q)ᵀ F_aero` (brief §5.4). Relative airflow at each element now includes full 6-DoF body motion: `v_rel = v_element(q, v) − v_air`. Wing forces remain per-term inspectable in the coupled run.

### Task 5 — Actuation stages 2–3 (`actuation/`)

- `profile_limited.py`: rate- and acceleration-limited abdomen profiles (stage 2, brief §6.4).
- `torque_driven.py`: torque-driven joint with damping, joint limits, saturation (stage 3); records commanded/actual position, velocity, torque, saturation flag per step.
- Stage 4 (servo dynamics + power) remains deferred; schema fields exist, behaviour raises `NotImplementedError` with a clear message if selected.

### Task 6 — Mechanism-isolation switches (brief §6.6)

First-class experiment-schema flags, one YAML per case under `experiments/m3_isolation/`:

| Flag | Effect |
|---|---|
| `mass_shift_enabled: false` | CoM/inertia frozen at neutral (baseline) |
| `inertial_coupling_enabled: false` | geometry-only CoG shift (quasi-static lever arms) |
| both true | full inertial coupling |
| `abdomen_aero_enabled` | stub — raises until Level-3 model exists |
| all | combined prediction |

### Task 7 — `simulation/runner.py`

`solve_ivp` wrapper: integrator/tolerances/`max_step` from `SolverSpec`; event hooks; NaN / excessive-motion / solver-failure guard ⇒ hard stop, record `FAILED`, failing state dumped (action plan §3 item 5).

### Task 8 — `experiment/record.py`

Full save/load: config YAML, parameter provenance, solver settings, `git rev-parse HEAD`, RNG seed, outputs `.npz` + metadata JSON, output schema version. `fwmav-sim rerun <record_dir>` reproduces the run (brief §9 step 11).

---

## 3.4 Test inventory

**Analytical (§11.2):** planar restriction of the full model reproduces M2 reduced-model trajectories to `< 1e-6` (`test_planar_limit_recovers_reduced_model`) — the key regression; free-body gravity moment about instantaneous CoM ≡ 0; aero-off free flight ⇒ linear momentum follows gravity exactly, angular momentum conserved to solver tolerance; zero-abdomen-motion baseline matches fixed-body results.

**Numerical (§11.3):** quaternion norm guard; timestep refinement does not change declared metrics within documented tolerance (table in report); save/reload + headless `rerun` ⇒ outputs bit-identical for fixed-step, within solver tolerance for adaptive (the rule itself documented in the record); solver-tolerance sensitivity sweep.

**Failure-injection:** NaN-forcing config ⇒ run stops, record marked `FAILED` (asserted in `tests/numerical/test_failure_behaviour.py`).

**Unit (§11.1):** spatial transform algebra vs 4×4 homogeneous equivalents; CRBA symmetry/positive-definiteness; RNEA bias vs Lagrangian computation on random states (reuse M2's SymPy machinery extended to 3D); actuator limit clamping.

## 3.5 Acceptance commands

```bash
pytest tests/ -q
fwmav-sim run experiments/m3_free_body_hover.yaml
fwmav-sim run experiments/m3_isolation/geometry_only.yaml
fwmav-sim run experiments/m3_isolation/full_coupling.yaml
fwmav-sim rerun outputs/<previous_run>
```

## 3.6 Milestone report

`docs/milestone_reports/M3.md`: DR-001 outcome summary; planar-limit regression numbers; conservation test numbers; isolation-case comparison table (brief §6.6 — more informative than one fitted curve); lever-arm and joint-torque time-series examples; timestep-refinement table; demonstrated failure-stop behaviour; repeatability evidence.

## 3.7 Explicit non-goals for this stage

Abdomen aerodynamic model (flag + stub only); servo dynamics with power estimates (stage 4); experimental-data comparison (M4); controllers (interface stub only); trim/Floquet analysis; UI.
