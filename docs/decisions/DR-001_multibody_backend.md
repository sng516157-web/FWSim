# DR-001 — Multibody dynamics backend: in-house spatial-vector algorithms

**Status:** confirmed 2026-11-03 (Route A). **Milestone of effect:** M3.

## Decision

Implement the articulated dynamics in-house using Featherstone spatial-vector
algorithms — Composite Rigid Body Algorithm for `M(q)`, Recursive
Newton–Euler for `C(q,v)v + G(q)` — restricted to the actual MVP topology:
one 6-DoF floating base plus up to 2 revolute abdomen joints (≈ 8 generalized
coordinates). NumPy only; no external dynamics engine.

## Rationale

- Drake pip wheels are Linux/macOS-only; Pinocchio pip wheels are Linux-only
  (Windows requires conda-forge). Neither is installable via pip on the
  target Windows machine (verified 2026-11, action plan §1).
- At 8 DoF the in-house implementation is ~300 lines of auditable NumPy with
  direct access to every term — required for the mechanism-isolation
  experiments of brief §6.6. This is far from the "general physics engine"
  brief §8.1 forbids.
- No black box between the equations and the validation tests.

## Cross-validation (not a runtime dependency)

An optional dev-only script may install Pinocchio from conda-forge in a
*separate* environment (or WSL2) and diff `M(q)`, `C`, `G` against the
in-house implementation on random configurations. Test asset only; never
imported by the package.

## Revisit trigger

Topology grows past 2 internal joints, closed-loop linkages appear, or
analytical derivatives (trim, Floquet) become the bottleneck.

Full record: `FWMAV_Milestone_Action_Plan.md` §1.2.
