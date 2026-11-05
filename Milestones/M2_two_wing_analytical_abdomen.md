# Milestone 2 — Two-Wing Vehicle and Analytical Abdomen Model

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — especially §5.4, §6.1–6.3, §10 M2, §11, §12
> - `../FWMAV_Milestone_Action_Plan.md` — stack §1, conventions §3
> - `docs/conventions/` — frozen M0 output; `docs/conventions/reduced_model_spec.md` — the EOM you must implement
>
> Conflicts: the brief wins; stop and flag. Route A: pure Python only, no Drake/Pinocchio/conda runtime deps.

**Goal:** two-wing configuration + longitudinal thorax–abdomen analytical model with verified instantaneous CoM/inertia and passive/actuated pendulum behaviour (brief §10 M2).
**Estimated effort:** 1.5–2 weeks.
**Prerequisite:** M1 accepted (working fixed-body aero, averaging, experiment records).

---

## 2.1 Ground rules for this stage

- This is the project's **trust anchor**: the reduced analytical model derived here is the regression reference for the M3 nonlinear model. Correctness beats speed everywhere.
- CoM and inertia derive from actual body geometry and configuration, never from a visualization marker (brief §6.2).
- Gravity produces **no moment about the instantaneous CoM** in free body; the suspended string is a **separate constraint model**, never conflated with free flight (brief §6.3). Pendulum agreement is not free-flight validation, and the milestone report must say so (brief §10 M2).
- Abdomen actuation this stage: **prescribed angle only** (stage 1 of brief §6.4).

## 2.2 Locked stack for this stage

Previous stack plus **SymPy ≥ 1.12** (dev-only, in `derivations/`). NumPy implementation must remain SymPy-free at runtime.

---

## 2.3 Tasks (in order)

### Task 1 — `vehicle/definition.py`

Load the full vehicle from YAML: thorax body, abdomen body (concentrated mass now; schema already allows distributed), two wings, joint definitions (abdomen pitch exercised now; roll parsed and validated but not exercised until M3). Validate: masses positive, inertia tensors symmetric positive-definite, joint axes unit vectors, left/right wing symmetry declared explicitly.

### Task 2 — `dynamics/mass_properties.py`

Implement brief §6.2 verbatim:

```python
def com(q: np.ndarray, vehicle: VehicleDefinition) -> np.ndarray: ...
def inertia_about_com(q: np.ndarray, vehicle: VehicleDefinition) -> np.ndarray: ...
```

`r_CoM(q) = Σ mᵢrᵢ(q) / Σ mᵢ`; `I_CoM(q) = Σ [RᵢIᵢRᵢᵀ + mᵢ(‖dᵢ‖²I − dᵢdᵢᵀ)]`, `dᵢ = rᵢ − r_CoM`. Both must respond to abdomen angle changes (time-varying mass properties, brief §3.4).

Tests (`tests/analytical/test_mass_properties.py`): three hand-computed configurations — (a) symmetric neutral, (b) abdomen deflected 30°, (c) point-mass limit — agreement to `< 1e-12`; inertia tensor symmetric PD for 100 random configurations; **free-body gravity moment about the instantaneous CoM ≡ 0** to machine precision (`test_gravity_moment_about_com_zero`).

### Task 3 — `derivations/reduced_longitudinal.py` (SymPy, dev-only)

Derive the two-mass longitudinal EOM (states `[x, z, θ, α_abd]` per `reduced_model_spec.md`) via Euler–Lagrange: kinetic/potential energy of both masses, abdomen joint at its documented location, gravity, generalized forces slot for cycle-averaged wing forces. Export: (a) human-readable LaTeX into the spec doc, (b) a generated numeric reference module `dynamics/_reduced_reference.py` via `sympy.lambdify`. Rerunning the script must regenerate and diff-check the reference.

### Task 4 — `dynamics/reduced_longitudinal.py` (NumPy runtime)

Hand-written implementation of the same EOM behind a `ReducedAnalyticalModel` class exposing `rhs(x, t, forces) -> ẋ`. Cross-check test: against the SymPy-lambdified reference at 100 random states, relative error `< 1e-10` (`tests/analytical/test_reduced_vs_sympy.py`).

### Task 5 — `constraints/suspended.py`

Suspended-string pivot as an explicit `TestConstraint` (brief §5.4, §6.3): pivot location, string length, pendulum DOF reduction. Clearly a different plant from free body — separate class, separate experiment configs.

### Task 6 — `actuation/prescribed_abdomen.py`

Prescribed α_abd(t) profiles (step, ramp, sinusoid); record commanded vs actual angle/velocity in the run record (brief §6.4 stage 1).

### Task 7 — Cycle-averaged wing-force injection

The reduced model consumes wingbeat-averaged forces **derived from an M1 time-resolved run** loaded as data (brief §3.3) — not from an independent formula. Loader validates that the averaged file's vehicle/kinematics config matches the current vehicle (warn + record on mismatch).

### Task 8 — Runnable experiments

`experiments/m2_passive_pendulum.yaml` (release from small initial angle, no actuation) and `experiments/m2_actuated_pendulum.yaml` (prescribed abdomen motion under suspension), each producing instantaneous + averaged outputs and plots.

---

## 2.4 Test inventory

**Unit (§11.1):** CoM/parallel-axis (Task 2); joint-limit clamping; profile generation derivatives.

**Analytical (§11.2):** gravity moment about CoM ≡ 0; passive pendulum small-oscillation frequency vs analytical `ω = sqrt(m·g·d / I)` to `< 1e-6` relative (`test_pendulum_frequency`); small-angle linearization vs numerical Jacobian of the nonlinear rhs (`test_linearization_matches_jacobian`); symmetric two-wing flapping ⇒ zero net roll/yaw moment (`test_symmetric_flapping_symmetry`); NumPy vs SymPy reference (Task 4); **RL-sandbox foil tests** (§2.4a).

**Numerical (§11.3):** frictionless passive pendulum energy drift `< 1e-8` per period (DOP853, rtol 1e-9, atol 1e-12); timestep refinement table (steps-per-wingbeat sweep on the actuated case); quaternion/attitude normalization n/a here (planar), noted as deferred to M3.

### 2.4a Foil test — RL-sandbox dynamics kernel (added 2026-11-03)

**Purpose:** cross-check the M2 dynamics kernel in its single-rigid-body limit against Steven's 2D RL sandbox (`github.com/sng516157-web/2D-RL-Control-for-a-flapping-wing-drone`; local copies under `docs/reference/rl_sandbox/`; see action plan §8.1). This validates the *kernel* (mass/inertia/force-law/lever-arm assembly) against an independently written, working implementation. It validates **nothing about aerodynamics** — the sandbox's `T = k·f` linear thrust law is adopted here only as a synthetic cycle-averaged force law, labelled `source: estimated`, never presented as physics.

**Configuration:** `experiments/m2_foil_rl_sandbox.yaml` —
- Abdomen joint **locked at α_abd = 0**, abdomen mass zero (or coincident with thorax CoM): two-mass model degenerates to one rigid body.
- Force law injected via the Task 7 cycle-averaged interface as a synthetic table: per-wing `T = k_thrust · f`, total along body "up" (−z_B), application points at `x = ±arm` on the body x-axis (fore/aft — the sandbox's in-plane "left/right" maps to the longitudinal plane; the sandbox's "yaw" is in-plane rotation, i.e., pitch here).
- Unsteady/noise torque terms **disabled** (`unsteady_rel_amp = 0`, `unsteady_noise_scale = 0`); documented in the run record.
- Damping included: `lin_damp = 0.05 N·s/m`, `ang_damp = 0.001 N·m·s/rad`. **Requirement this imposes on Tasks 3–4:** the reduced model spec and EOM must carry optional linear/angular damping slots.

**Parameter mapping (sandbox `PARAMS` → FWSim schema):**

| Sandbox | Value | FWSim target |
|---|---|---|
| `m` | 0.05 kg | thorax `BodySpec.mass` (abdomen mass 0, locked) |
| `I` | 2e-4 kg·m² | thorax pitch inertia `I_yy` about CoM |
| `arm` | 0.1 m | thrust application points `x = ±0.1 m` |
| `k_thrust` | 0.02 N/Hz | synthetic averaged-force table coefficient |
| `lin_damp` / `ang_damp` | 0.05 / 0.001 | reduced-model damping slots |
| `dt = 0.02`, Euler | — | **not replicated** — FWSim keeps its own solver; comparison tolerance accounts for it |

**Frame mapping (verify by test, do not assume):** sandbox world `(x, y-up)` ↔ FWSim `(x, −z)`; sandbox `θ` (0 = body-up, positive tilts body-up toward +x) ≡ FWSim pitch θ (0 = level, positive nose-up); sandbox thrust `(T sinθ, T cosθ) − gravity` ↔ FWSim `F_x = T sinθ`, `F_z = mg − T cosθ`; `τ = arm·(T_R − T_L)` ↔ pitch moment `M_y = arm·(T_front − T_back)` with `T_R ↔ T_front` — confirm sign via the mirror/reversal convention tests before trusting the foil.

**Tests and pass criteria:**

1. `test_foil_rhs_equivalence` — at 100 random states/inputs, FWSim reduced-model accelerations match the sandbox's closed-form dynamics to `< 1e-12` (sandbox dynamics are analytic; compute expectations by hand formula, not by running sandbox code).
2. `test_foil_hover_equilibrium` — at `base_f = m·g/(2·k_thrust) = 12.2625 Hz`, `freq_delta = 0`, zero perturbation: equilibrium held to solver tolerance over 10 s.
3. `test_foil_pitch_response` — small perturbation `θ₀ = 0.05 rad`, no control: FWSim trajectory matches the linearized analytical prediction of the *sandbox* equations to `< 1e-6` relative over 2 s. (Compare against the closed-form linearization, not the sandbox's Euler trajectory — Euler at dt = 0.02 is itself an approximation.)

Report foil results in the M2 milestone report with the caveat that this is a kernel cross-check against a toy model, not physical validation.

## 2.5 Acceptance commands

```bash
python derivations/reduced_longitudinal.py     # regenerates reference, asserts match
pytest tests/unit tests/analytical tests/numerical -q
fwmav-sim run experiments/m2_passive_pendulum.yaml
fwmav-sim run experiments/m2_actuated_pendulum.yaml
fwmav-sim run experiments/m2_foil_rl_sandbox.yaml
```

## 2.6 Milestone report

`docs/milestone_reports/M2.md` must include: CoM/inertia verification table; pendulum frequency comparison; energy-drift numbers; the **explicit statement** that suspended-pendulum agreement is not free-flight validation (brief §10 M2, §12); and `data_status` of the reference vehicle — if Leng et al. original parameters/time series are still unavailable, numerical reproduction of the paper stays `PENDING` with exactly that label.

## 2.7 Explicit non-goals for this stage

Six-DoF motion; abdomen roll joint motion; torque-driven or servo abdomen dynamics; time-resolved aero coupled into the dynamics (cycle-averaged injection only); abdomen aerodynamics; experimental comparison.
