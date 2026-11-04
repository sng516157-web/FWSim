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
| A-008 | Dickinson 1999 lift/drag fits are evaluated on the **flipped-wing-equivalent incidence**: flight-convention α is folded to |α| ∈ [0°, 90°] with odd (C_L) / even (C_D) symmetry, and the sign of α tells which wing surface faces the flow. The published fit's α > 90° branch (non-flipped TE-first plate) is not used, because the wing flips at stroke reversal (Sane & Dickinson convention; see the "opposite sign in upstroke and downstroke" usage). | `aero/quasi_steady.py` `cl_cd` (M1; corrected post-M1, DR-003). | If the paper's α convention is found to differ, translational forces at |α| > 45° are affected. Verify against Dickinson et al. 1999 when the paper text is available (pending). | M1.1 |

## How to cite

```python
def element_forces(...):
    """... Assumptions: A-001 (rigid wing), A-003 (quasi-steady), A-006 (no inflow model)."""
```

## Pending / explicitly not assumed

- Abdomen aerodynamics (brief §6.5 Level 3) are **not** assumed negligible in
  general; they are simply unmodelled until Level 3. Mechanism-isolation runs
  (brief §6.6) must keep the `abdomen_aero_enabled` flag visible.
- Suspended-string experiments have pendulum dynamics and a physical pivot;
  they are **not** assumed equivalent to free flight (brief §6.3) — they get
  their own constraint model (M2) rather than an assumption entry.
