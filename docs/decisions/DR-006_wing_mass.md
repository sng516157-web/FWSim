# DR-006 — Wing structural mass: kinematically driven appendages folded into M(q)

**Status: accepted (post-M5 item 1).** Supersedes the "wings are massless"
recorded limitation of the M2/M3 reports.

## Context

M1–M5 treated wings as massless force transmitters. For FWMAVs the wings are
a significant mass fraction (AirPulse: wing assemblies ≈ 38% of flight-ready
mass, Gu et al. 2026), and their prescribed flapping motion produces inertial
reaction forces on the body of order `m_wing · phi_amp · (2 pi f)^2 · R` —
not negligible against body weight.

## Decision

Wings with `WingSpec.mass` set are **kinematically driven rigid appendages**
(A-004, A-010): no new dynamic DoFs. Per wing, at each instant:

- the wing's spatial inertia about the base origin is added to the base block
  of the mass matrix (`M_add`), and
- the velocity-level / prescribed-acceleration / gravity part of the wing
  reaction wrench is added to the right-hand side (`Q_bias`).

The `u_dot`-dependent part of the wing reaction is thereby on the left-hand
side — the consistent treatment of the algebraic loop already noted for fluid
added mass in A-009. Mass properties derive analytically from the planform as
a uniform-areal-density thin plate, including the `I_xy` product of inertia
for tapered planforms with an offset pitch axis. All kinematic rates come
from the frozen transform chain; the chain was extended with the analytic
child-frame angular acceleration (`wing_chain_with_alpha`).

Implementation: `dynamics/wing_inertia.py`; consumed by
`simulation/free_body.py` (base block via `forward_dynamics(m_extra,
rhs_extra)`) and `simulation/fixed_body.py` (`F_inertial` / `M_inertial` /
`F_sensor` channels — what a force balance measures on top of the aero
force). Isolation switch `wing_inertia_enabled` (brief §6.6) in
`IsolationSpec` / `IsolationFlagsSpec`.

## Consequences

- **Backward compatible:** wings without `mass` are massless exactly as
  before (M1–M5 records unchanged; verified by the reproducibility tests).
- The Level-1 reduced longitudinal model is unchanged: it consumes
  cycle-averaged forces (A-007) and the inertial force of a periodic stroke
  is zero-mean beyond the wing weight.
- **AirPulse finding:** the paper's 38% wing-assembly mass is inconsistent
  with a uniform-plate wing at the measured Table A1 inertia (consistency
  bound 0.82 g/wing; assembly mass must sit in the root mechanism). The
  `airpulse_wingmass.yaml` variant carries a 0.5 g/wing membrane estimate
  with body mass/inertia adjusted so totals still match Table A1 exactly
  (derivations/airpulse_wingmass.py).
- **AirPulse rig gap:** wing inertia at the membrane estimate adds ~2.3%
  oscillatory root force and zero change to cycle-mean coefficients — the
  M4 C_F gap is NOT wing-mass bookkeeping; it remains aero-model/regime.
