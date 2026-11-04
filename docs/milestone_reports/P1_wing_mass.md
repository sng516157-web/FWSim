# P1 — Wing structural mass in the dynamics (post-M5 queued item 1)

- **Date:** 2026-11-04
- **Software revision:** `98ef8b0` + this change
- **Environment:** Python 3.14.7, Windows (Git Bash), numpy/scipy per `pip freeze`

Post-roadmap item: wings were massless force transmitters through M5 (recorded
limitation, M2/M3 reports; the largest model-fidelity gap, and the candidate
structural explanation for the M4 AirPulse comparison gap).

## 1. Implemented capabilities

| Capability | Where | Tested |
|---|---|---|
| Optional `WingSpec.mass` (schema; None = massless, backward compatible) | `vehicle/schema.py` | yes |
| Planform-derived thin-plate mass properties (uniform areal density, all three planforms, incl. `I_xy` for tapered wings with offset pitch axis) | `dynamics/wing_inertia.py` | yes (analytic + quadrature + point-mass assembly) |
| Analytic child-frame angular acceleration of the frozen wing chain | `kinematics/blade_element.py` `wing_chain_with_alpha` | yes (finite-difference) |
| Wing spatial inertia folded into the base mass matrix + reaction wrench on the rhs (DR-006; no new DoFs) | `simulation/free_body.py`, `dynamics/spatial.py` `forward_dynamics(m_extra, rhs_extra)` | yes (momentum conservation) |
| Fixed-body rig inertial channels `F_inertial` / `M_inertial` / `F_sensor` (what a force balance measures on top of aero) | `simulation/fixed_body.py` | yes |
| Isolation switch `wing_inertia_enabled` (experiment + plant) | `experiment/config.py`, `simulation/free_body.py` | yes |
| AirPulse wing-mass variant vehicle + rig experiment | `reference_vehicles/airpulse_wingmass.yaml`, `experiments/m4_airpulse_rig_wingmass.yaml`, `derivations/airpulse_wingmass.py` | yes (run recorded) |

## 2. Tests run and actual results

| Check | Command | Result |
|---|---|---|
| wing-inertia suite | `pytest tests/analytical/test_wing_inertia.py -q` | **11 passed** |
| planform properties vs closed form / quadrature | (in suite) | rectangular: exact (m R²/12 etc.); trapezoidal/elliptical: < 1e-6 relative |
| chain angular acceleration | (in suite) | matches finite difference of the chain omega to rtol 1e-6 |
| rigid-wing terms vs independent 400-element point-mass assembly | (in suite) | mass matrix and bias agree to ~2e-6 relative (element-midpoint discretization noise) |
| free body, aero off, wings flapping (2 × 4 g wings on a 20 g body) | (in suite) | total linear momentum follows `m_tot·g·t` to **1e-7**; angular momentum about the world origin follows the integrated gravity moment to **2e-7** over 0.4 s |
| fixed-body rig, 4 g wing at 10 Hz | (in suite) | cycle-mean inertial force = wing weight exactly (m·g, atol 2e-4); fluctuating part zero-mean, amplitude 2.1 N-scale |
| backward compatibility | `pytest tests -q` | **110 passed / 4 skipped** — M1–M5 records unchanged (massless default) |
| AirPulse rig rerun with wing mass | `fwmav-sim run experiments/m4_airpulse_rig_wingmass.yaml` | record `outputs/20261104_204313_m4_airpulse_rig_wingmass`; aero channels bit-identical to the massless run; mean inertial force = 4.905e-3 N = wing weight; peak inertial 0.049 N vs peak aero 2.13 N (~2.3%) |

## 3. Findings

- **The M4 AirPulse gap is not wing-mass bookkeeping.** Wing inertia on a
  fixed rig is oscillatory; its cycle mean beyond the wing's own weight is
  exactly zero (periodicity). At the 0.5 g membrane estimate it adds ~2.3%
  oscillatory root force; even at the full 4.88 g assembly mass it would be
  ~22% oscillatory, still zero-mean. The C_Fx/C_Fz gap (ours 0.071/1.408 vs
  paper 6.084/4.780) remains aerodynamic-model/regime (Dickinson Robofly fits
  at Re ~ 3e4), as recorded in M4.
- **AirPulse data inconsistency (new):** "wing assemblies ≈ 38% of
  flight-ready mass" (4.88 g/wing) is incompatible with the Table A1
  whole-robot inertia under a uniform-plate wing model — the uniform-plate
  inertia at neutral exceeds the measured whole-robot value. Consistency
  bound: ≤ 0.82 g/wing. The assembly mass must be concentrated in the root
  mechanism. `airpulse_wingmass.yaml` therefore carries a **0.5 g/wing
  membrane estimate** (provenance: estimated); body mass/inertia = Table A1
  minus the neutral-config wing model, so total mass and neutral inertia
  remain exactly the paper's values (derivations/airpulse_wingmass.py,
  remainder eigenvalues [1.96e-5, 1.26e-4, 1.87e-4] > 0).

## 4. Known limitations

- Uniform-areal-density plate (A-010): real wing mass is root-concentrated
  (spar). A measured wing mass distribution would replace the planform model.
- Wing mass properties derive from the *planform* approximation (AirPulse:
  rectangular at MAC), inheriting that approximation's error.
- The Level-1 reduced longitudinal model is unchanged by design (A-007).
- Wing flapping actuator torques are still not modelled (A-004): the wing
  reaction wrench is applied to the body, but no actuator model checks
  feasibility.

## 5. Reproduction

```bash
pip install -e .[dev]
pytest tests/analytical/test_wing_inertia.py -q
python derivations/airpulse_wingmass.py
fwmav-sim run experiments/m4_airpulse_rig_wingmass.yaml
```

## 6. Acceptance self-check

Queued item: "wing-mass bodies in dynamics (top fidelity gap — would test
whether the AirPulse gap is structural)".

- Wing mass in the Level-2 dynamics, exactly coupled (no new DoFs): **PASS**
  (momentum conservation to 1e-7 / 2e-7 with flapping massive wings).
- AirPulse-gap question answered: **PASS — not structural (not wing mass)**;
  the gap is aero-model/regime. A new data inconsistency in the paper's mass
  breakdown was identified and quantified instead.
