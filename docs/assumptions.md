# Assumption Register

**Status: living register, seeded in M0.** Rule (stage file Task 6): code that
relies on an assumption cites its ID (e.g. `A-003`) in the docstring of the
implementing function/class. Removing or weakening an assumption requires a
code change, a test change, and an edit to this register in the same commit.

| ID | Assumption | Scope / where it applies | Consequence if violated | Introduced |
|---|---|---|---|---|
| A-001 | Wings are rigid bodies. | All aerodynamic and kinematic code (M1+). | Flexible-wing loads, spanwise twist and their effect on AoA are absent (brief §5.3, fidelity level 1 only). | M0 |
| A-002 | No wake capture and no clap-and-fling. | Default quasi-steady aerodynamic model (brief §5.1). | Force peaks near stroke reversal and wake-recapture energy recovery are not represented. | M0 |
| A-003 | Quasi-steady regime: instantaneous forces depend on instantaneous kinematic state (plus local acceleration for added mass), not on history. | Default aerodynamic model (M1). | Circulatory lag and unsteady wake memory effects are excluded; validity limited to the regime of the fitted coefficients (Re, advance ratio). | M0 |
| A-004 | Wing kinematics are prescribed (no aeroelastic or drivetrain feedback on wing motion). | M1–M2; relaxed only by later actuator/mechanism milestones. | Wingbeat does not respond to aerodynamic or inertial loading; actuator feasibility must be checked separately (brief §5.6). | M0 |
| A-005 | No wing deformation: planform, chord and area are time-invariant. | Blade-element discretization (M1+). | Morphing-wing effects excluded; deformation enters only as an imported/prescribed field in later fidelity levels. | M0 |
| A-006 | No inflow model beyond relative velocity: each blade element sees `v_rel = v_element − v_air`; no induced-velocity field, no ground effect. | Environment + aero coupling (brief §5.4). | Rotor-style induced inflow, wake contraction and ground effect are absent. | M0 |
| A-007 | Cycle-averaged forces are used **only** in the reduced longitudinal model, derived from a time-resolved run of the full model (brief §3.3). | `reduced_longitudinal` model (M2). | Wingbeat-scale oscillatory dynamics are invisible to the reduced model; it must never be used for within-wingbeat statements. | M0 |
| A-009 | In the free-body model, the added-mass term uses velocity-level and flapping accelerations only — the body's translational/angular accelerations are excluded because they depend on the forces being computed (algebraic loop). | `kinematics/blade_element.py` `element_state_free_body` (M3). | The aero force misses the body-acceleration contribution to added mass; small while wing added mass ≪ body inertia. A consistent treatment would fold fluid added mass into M(q) — deferred. (The *structural* wing mass got exactly that treatment in DR-006, post-M5.) | M3 |
| A-010 | Wing structural mass, when configured (`WingSpec.mass`), is modelled as a thin flat plate of uniform areal density following the prescribed wing kinematics exactly (A-001, A-004); mass properties derive analytically from the planform. The wing's inertial reaction is folded into the base block of M(q) and the right-hand side — no new DoFs (DR-006). | `dynamics/wing_inertia.py`; consumers: `simulation/free_body.py`, `simulation/fixed_body.py` (P1). | Real wings are not uniform-density plates (spar root-concentrated); the Level-1 reduced model is unchanged (cycle-averaged forces only, A-007 — inertial forces of a periodic stroke are zero-mean beyond the wing weight). | P1 |

## How to cite

```python
def element_forces(...):
    """... Assumptions: A-001 (rigid wing), A-003 (quasi-steady), A-006 (no inflow model)."""
```

## Pending / explicitly not assumed

- Abdomen aerodynamics (brief §6.5 Level 3) are **modelled** as of post-M5
  item 4 (`aero/abdomen.py`: flat plate with own CoP and Dickinson-form fits,
  applied at the abdomen body; `abdomen_aero` block in the vehicle YAML +
  `abdomen_aero_enabled` isolation flag). Abdomen–wing wake interaction is
  NOT modelled (A-002 extends to the abdomen); coefficients are placeholders
  until experimentally identified (brief §6.5 Level 3 correction path).
- Suspended-string experiments have pendulum dynamics and a physical pivot;
  they are **not** assumed equivalent to free flight (brief §6.3) — they get
  their own constraint model (M2) rather than an assumption entry.
