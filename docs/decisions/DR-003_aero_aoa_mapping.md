# DR-003 — Angle-of-attack mapping into the Dickinson 1999 fits

**Status:** decided 2026-11-04, post-M1 correction. **Milestone of effect:** M1+ (all aero).

## Problem found

The first hover runs mapped the flight-convention signed α onto the Dickinson
et al. (1999) fits folded into [0°, 180°]. At mid-backstroke (trailing edge
first after the wing flip) this evaluates the fit at α = 135°, where the
published C_L is negative — producing **downforce on the upstroke** (measured
in the stored run: mid-backstroke F_z = +0.101 N, i.e. downward). Reported by
Steven; confirmed from the run record.

## Decision

The fits are evaluated on the **flipped-wing-equivalent incidence**: fold
|α| into [0°, 90°], apply odd symmetry to C_L and even to C_D, and carry the
sign via the flight-convention α (which surface faces the flow). The published
fit's α > 90° branch is unused: a flapping wing flips at reversal, so both
half-strokes present the same surface at the same incidence (consistent with
the Sane & Dickinson usage where α has opposite sign in up- and downstroke,
and with the fit being ~sin(2α)-odd in structure). Assumption register: A-008.

## Evidence

- Anti-regression test `test_hover_kinematics_lift_up_both_half_strokes`
  (tests/analytical/test_fixed_body.py): both half-strokes produce upward
  force; drag opposes stroke motion on both.
- Post-fix hover run roughly doubles cycle-mean lift (see M1 report erratum).

## Caveat (brief §16)

The exact α convention of Dickinson et al. 1999 should be verified against
the paper text when available; until then this mapping is a documented
modelling decision pinned by physical requirements (lift up on both
half-strokes for hover kinematics), not a claimed transcription of the paper.
