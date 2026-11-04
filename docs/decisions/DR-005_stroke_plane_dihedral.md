# DR-005 — Per-wing stroke-plane dihedral (stroke-plane roll tilt)

**Status:** decided 2026-11-04 (M4 prep, AirPulse reference vehicle).
**Milestone of effect:** all aero/kinematics code paths.

## Decision

`WingSpec` gains an optional `dihedral_angle` γ (rad, default 0). The
stroke-plane frame of each wing becomes

```text
R_B_SP_wing = Rx(σ γ) · Ry(β),   σ = −1 (right), +1 (left)
```

i.e. γ > 0 raises both wingtips (V-dihedral). Applied in the transform chain
before β; γ = 0 recovers the M0-frozen chain exactly. `frames.md` §1.4 amended
in the same commit, per the frozen-convention change rule.

## Rationale

The AirPulse reference platform (arXiv:2602.06811) has a fixed 50° V-dihedral
— too large to ignore or silently approximate. DelFly platforms also carry
dihedral (DelFly II: 12°, servo-adjustable on the Nimble). The alternative
(folding dihedral into β) conflates roll and pitch tilts and breaks the
left/right mirror structure; a dedicated per-wing angle with mirrored signs
keeps `F_L = S·F_R` exact (test added).

## Consequences

- Mirrored-dihedral symmetry test added (`test_blade_element.py`).
- Default γ = 0 everywhere else: all M0–M3 results are unaffected.
