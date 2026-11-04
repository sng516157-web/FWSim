# DR-001 Outcome — In-house spatial-vector dynamics confirmed (M3.0 spike)

**Date:** 2026-11-04. **Decision upheld:** in-house NumPy implementation, no
external dynamics engine in the runtime path.

## What was implemented

`dynamics/spatial.py`: Featherstone-style spatial algebra restricted to the
MVP topology (6-DoF floating base + abdomen pitch + abdomen roll, n = 8):

- spatial motion/force transforms, spatial inertia about off-CoM frame origins;
- `crba(q)` — composite-rigid-body M(q) via the body-Jacobian assembly
  (algebraically identical to CRBA; at n = 8 the Jacobian form is the same
  computation written flat);
- `rnea(q, u, qdd)` — outward/inward recursion for bias forces and internal
  joint reaction torques;
- `forward_dynamics` — direct 8×8 solve (no ABA, per DR-001);
- `body_jacobians`, `forward_kinematics`, `com` for aero mapping and
  lever-arm outputs.

## Validation numbers (all in the committed test suite)

| Check | Result | Where |
|---|---|---|
| M(q) planar restriction vs M2 SymPy reference | max abs diff **3.5e-18** (100 random configs) | `test_crba_planar_restriction_matches_sympy` |
| M(q) symmetry + positive-definiteness | exact, 100 random configs | `test_crba_symmetric_positive_definite` |
| RNEA bias vs energy-derived Jacobian bias | **5.8e-14** | derivations/probe_bias.py |
| RNEA bias planar restriction vs M2 (with coordinate-map correction) | < 1e-10 | `test_rnea_bias_planar_matches_sympy` |
| Planar trajectory regression vs M2 reduced model | < 1e-6 over 2 s (x, z, R(θ), α) | `test_planar_limit_recovers_reduced_model` |
| Gravity-free conservation over 1 s tumble | Δp_lin 4e-16 · ΔKE 2e-13 (rel) · ΔL < 1e-9 | `test_gravity_free_momentum_and_energy_conservation` |
| Gravity free fall: p(t) = p0 + m·g·t | < 1e-9 | `test_gravity_linear_momentum_follows_free_fall` |
| Clamped chain ≡ composite single rigid body | identical trajectory to 8+ digits | `test_clamped_chain_equals_composite_single_body` |

## Convention deviation discovered during the spike (important)

The first RNEA draft mixed two spatial-acceleration conventions: it used the
componentwise-derivative acceleration (as the CRBA/Jacobian path implies) but
kept Featherstone's `V × S q̇` Coriolis term, which belongs to the
spatial-acceleration convention. The error showed up only in base×joint
cross-coupling (probe: `derivations/probe_bias3.py`, max mismatch 4e-3 in the
mixed case; momentum drift 24% of L). The committed recursion uses the
componentwise convention with the analytic `dX_up/dt · V_parent` term
(`_dX_up_dt`), verified against the energy-derived Jacobian bias to 5.8e-14
and against conservation laws above.

## Pinocchio cross-check

Skipped: conda is not available on this machine (`conda: command not found`,
checked 2026-11-04). Per DR-001 the Pinocchio diff was optional (test asset
only). The SymPy + conservation + composite-equivalence coverage substitutes
for it. Revisit trigger unchanged (topology growth, closed loops, trim/
Floquet derivative needs).

## Massless carrier bodies

The serial pitch→roll abdomen chain uses a massless intermediate body
(`abdomen_roll_carrier`), which required a schema extension:
`BodySpec.massless: true` (zero mass and zero inertia enforced).
