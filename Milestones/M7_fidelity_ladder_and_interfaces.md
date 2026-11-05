# Milestone 7 — Fidelity Ladder & Learning Interfaces

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — §3.1, §5.1, §5.8, §6.5, §11
> - `../docs/literature/open_source_sim_survey.md` — the M7 method source
> - M0–M5 + P1–P4 reports, M6 report
>
> Conflicts: the brief wins; stop and flag. Route A: pure Python + PySide6.

**Goal:** (a) a validated UVLM aero tier above the quasi-steady model, reached by WRAPPING PteraSoftware (pip, validated, maintained) rather than writing a home solver; (b) passive torsional wing-pitch compliance — the AirPulse model's largest known input error; (c) helicopter-style control channels + a Gymnasium RL wrapper so the platform is usable as a control testbed; (d) the fidelity ladder made explicit and visible in records.
**Prerequisite:** M6 accepted.

**Out of scope (recorded):** home-written UVLM (wrap instead); rVPM/CFD (FLOWUnsteady tier — revisit if UVLM proves insufficient); MuJoCo migration (rejected: our dynamics core is verified and provenance-integrated); aeroelastic structural dynamics beyond torsional pitch.

---

## 7.1 Ground rules

- Physics before interface; the adapter adds NO physics of its own — every force it returns comes from PteraSoftware's solver, recorded as such in the record's validity section (`aero backend: pterasoftware-uvlm (wrapped)`).
- New dependency `pterasoftware` follows the action-plan dependency protocol (§5): pip-installable, pinned in `pyproject.toml` as an optional extra `uvlm`, CI-guarded imports (`pytest.importorskip`) so the base install stays lean.
- Every new capability: tests first, provenance labels, validity-section visibility, docs update in the same commit.

## 7.2 Tasks (in order)

### Task 1 — UVLM spike + adapter backend (`aero/uvlm_backend.py`)

Spike first (one session, go/no-go recorded in the M7 report): can PteraSoftware's unsteady ring-UVLM be driven with (a) our prescribed wing kinematics, (b) our wing geometry, (c) a moving body frame? Then implement the adapter: an `AerodynamicModel`-compatible backend whose `element_forces` is replaced by per-wing UVLM force evaluation called from the runners (UVLM is not blade-element — the interface note below applies). Wake settings (fixed vs free, aging) exposed in the aero spec with defaults from Ptera's own validation case.

Interface note: UVLM solves per wing, not per element — the adapter consumes the same `ElementState`-producing kinematics but returns per-wing resultant + distribution; the runners map it into the existing channels (documented mapping, element distribution from UVLM panel loads).

### Task 2 — Cross-validation: UVLM vs quasi-steady vs AirPulse

Run the M4 AirPulse rig case through the UVLM backend and compare all three (UVLM vs our quasi-steady vs the paper's coefficients). The question that justifies the task: does the M4 C_F gap (ours 0.071/1.408 vs paper 6.084/4.780) move with wake physics included? Outcome recorded honestly either way (tuning into agreement remains forbidden). Also one Ptera-published validation case reproduced as a regression anchor.

### Task 3 — Passive torsional wing pitch (`dynamics/wing_pitch_compliance.py`)

Per-wing torsional spring-damper DOF replacing the prescribed pitch profile: pitch evolves under aero pitching moment + inertia + spring/damper, initial condition from static balance. This models AirPulse's passive feathering (and real compliant spars generally). Prescribed pitch remains the default (backward compatible); compliance is a per-wing flag with stiffness/damping parameters (provenance required). Coupling into both fixed-body and free-body runners; wing-mass inertia (DR-006) included in the pitch equation. Tests: static equilibrium vs hand computation, relaxation to prescribed-limit at infinite stiffness, energy check on free oscillation.

### Task 4 — Control channels + Gymnasium wrapper (`control/channels.py`, `control/gym_env.py`)

- Wing-beat modulation channels (flappy pattern): `[thrust, roll, pitch, yaw]` mapped to documented kinematic parameter modulations (amplitude scaling, differential stroke, mean offset, split-cycle asymmetry). A `ChannelMixer` translates channel commands to per-wing kinematics adjustments per beat.
- `FWSimEnv` (Gymnasium API): free-body hover task — observation = body state + phase, action = channel deltas, reward = hover keeping (documented reward, no tuning claims). Guards: gymnasium as optional extra; `importorskip` tests; one short training-free rollout test with the M6 LQR as a scripted policy.

### Task 5 — Fidelity ladder in records + docs

`aero.fidelity_tier` recorded in every run record (`quasi_steady` | `uvlm` | `quasi_steady+unsteady_terms`); validity panel shows it. User guide gains a "choosing the tier" section. Survey doc linked from README.

## 7.3 Test inventory

- Spike go/no-go recorded before Task 1 implementation (documented in M7 report §1).
- UVLM adapter: force sign conventions vs quasi-steady at overlapping regime (hover case: means within an honestly-recorded tolerance band — UVLM includes induced drag, so exact equality is NOT expected; the band and its rationale are documented).
- Ptera published validation case reproduced within their reported error bars.
- Compliance DOF: statics vs analytic, stiff-limit recovery of prescribed kinematics.
- Gym env: API conformance (gymnasium's `check_env`), rollout terminates, reward finite.
- Full suite green at every commit; M7 report with actual numbers.

## 7.4 Acceptance commands

```bash
pytest tests/ -q
fwmav-sim run experiments/m7_airpulse_rig_uvlm.yaml
python derivations/m7_uvlm_vs_quasisteady.py   # three-way C_F comparison
fwmav-sim run experiments/m7_passive_pitch_hover.yaml
python -c "import gymnasium; from fwmav_sim.control.gym_env import FWSimEnv; gymnasium.utils.env_checker.check_env(FWSimEnv())"
```
