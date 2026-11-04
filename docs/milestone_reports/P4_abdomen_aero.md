# P4 — Abdomen aerodynamics, Level 3 (post-M5 queued item 4; brief §6.5)

- **Date:** 2026-11-04
- **Software revision:** `612da8a` + this change
- **Environment:** Python 3.14.7, Windows (Git Bash)

The M3 isolation flag `abdomen_aero_enabled` was a documented stub that raised
`NotImplementedError`. It is now a real model.

## 1. Implemented capabilities

| Capability | Where | Tested |
|---|---|---|
| Flat-plate abdomen aero: own centre of pressure (`cp_offset` behind the joint along the abdomen axis), own Dickinson-form CL/CD coefficient set, spanwise (roll-axis) flow excluded | `aero/abdomen.py` | yes |
| `AbdomenAeroSpec` schema block on the vehicle (`area`, `cp_offset`, `coefficients`, all with provenance) | `vehicle/schema.py` | yes |
| Free-body plant wiring: wrench applied to the abdomen body via the body Jacobians (abdomen joint and thorax see the correct reaction); recorded channels `abdomen/F_aero`, `abdomen/alpha_aero` | `simulation/free_body.py`, `simulation/runner.py` | yes |
| Isolation semantics: enabling without a vehicle block or without an articulated abdomen raises a clear error | `simulation/free_body.py` | yes |
| Reference parameters (estimated placeholders) on the bio vehicle | `reference_vehicles/bio_abdomen_fwmav.yaml` | yes (run) |
| Isolation experiment | `experiments/m3_isolation/abdomen_aero.yaml` | yes (run recorded) |

## 2. Tests run and actual results

| Check | Command | Result |
|---|---|---|
| abdomen aero unit tests | `pytest tests/unit/test_abdomen_aero.py -q` | **7 passed** — force law vs hand-computed flat-plate values (drag along axis: `F = q·C_D(0)` downstream, lift `−q·C_L(0)` ẑ), zero-speed → zero, 90° degenerate → pure drag, spanwise flow excluded, CoP placement vs independent kinematics to 1e-12, reaction-torque presence/sign, 0.05 s free-body run finite |
| isolation flag semantics | `pytest tests/validation/test_mechanism_isolation.py -q` | old NotImplementedError test replaced: flag without vehicle spec raises `ValueError` |
| demo run | `fwmav-sim run experiments/m3_isolation/abdomen_aero.yaml` | record `outputs/20261104_211811_m3_iso_abdomen_aero`; peak abdomen aero force 2.9 mN (3 Hz, 0.3 rad swing of a 2.4e-4 m² plate), mean force ≈ (−0.08, 0, −0.28) mN |

## 3. Known limitations

- Abdomen–wing wake interaction is NOT modelled (the abdomen sees uniform
  air; A-002/A-006). For an abdomen directly behind flapping wings this is
  the largest error source.
- Coefficients are placeholders: the Dickinson 1999 functional form with
  Robofly-regime coefficients, labelled `estimated`; brief §6.5 Level 3
  calls for experimentally identified corrections — open until rig data.
- The plate normal is +z_abd (side profile). A real abdomen's 3D shape
  (dorsal/ventral asymmetry) is not represented.

## 4. Reproduction

```bash
pytest tests/unit/test_abdomen_aero.py -q
fwmav-sim run experiments/m3_isolation/abdomen_aero.yaml
```
