"""Optional reduced-order unsteady force terms (M6 Task 7; A-002 relaxed only
where a term is explicitly enabled in the vehicle YAML).

Both terms are pure functions of (wing, prescribed kinematics, t, element
geometry) — no hidden history state, so solvers may evaluate them in any
order.

**Wake capture** (phenomenon: Dickinson et al. 1999; reduced model lineage
Berman & Wang 2007, J. Exp. Biol. 210): after each stroke reversal the wing
re-encounters the vortex sheet shed at the previous reversal and produces a
transient force peak. Reduced form:

    F_wc = K_wc * (0.5 rho v_peak^2 c dr) * W(dt) * e_L
    W(dt) = exp(-dt/tau_decay) - exp(-dt/tau_rise),   dt = t - t_rev,last
    v_peak = phi_amp * omega * s  (peak flap speed of the element, previous
    half-stroke proxy — exact for sinusoidal stroke, no aero feedback)

tau_rise/decay are fractions of the half-stroke period (params, defaults
0.05 / 0.2 per the Berman & Wang peak timing). Direction: the current
half-stroke's lift direction e_L (the wake energizes the new stroke).

**Clap-and-fling** (phenomenon: Weiss-Fogh 1973; Ellington 1984 review):
when a mirrored wing pair's tips approach below a gap threshold, circulation
is augmented. Reduced form (left/right mirror pairs; X-wing fore-aft pairing
is a documented follow-up):

    F_cf = K_cf * (0.5 rho |v_rel|^2 c dr) * max(0, 1 - gap(t)/g_thresh) * e_L
    gap(t) = |r_tip,left - r_tip,right|  (tip mid-chord points, world)

Assumptions: A-001 (rigid wing), A-004 (prescribed kinematics — no wake
memory state), quasi-steady base model unchanged. These are ENGINEERING
reduced forms with an empirical gain, not resolved unsteady models; the gain
is labelled estimated and is a sysid fitting target (M6 Task 9).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import R_B_SP_wing
from fwmav_sim.kinematics.blade_element import dihedral_of, wing_chain
from fwmav_sim.kinematics.prescribed import wing_state

_HALF = (0.05, 0.2)  # default tau_rise / tau_decay as half-stroke fractions


def _reversal_clock(kin, t: float) -> tuple[float, float]:
    """(dt since last stroke reversal, half-stroke period)."""
    f = kin.flap_frequency.value
    delta_s = kin.stroke_phase.value if kin.stroke_phase is not None else 0.0
    w = 2.0 * np.pi * f
    # phi = amp cos(w t + delta_s) -> reversals at w t + delta_s = k pi
    phase = (w * t + delta_s) % (2.0 * np.pi)
    # distance to the most recent multiple of pi
    dt = (phase % np.pi) / w
    return dt, 0.5 / f


def wake_capture_force(wing, el, kin, t: float, e_L: np.ndarray, rho: float,
                       spec) -> np.ndarray:
    """Wake-capture force on one element; zero before the first reversal."""
    dt, half_T = _reversal_clock(kin, t)
    if dt <= 0.0:
        return np.zeros(3)
    tau_rise = spec.params.get("tau_rise_frac", _HALF[0]) * half_T
    tau_decay = spec.params.get("tau_decay_frac", _HALF[1]) * half_T
    window = np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)
    if window <= 0.0:
        return np.zeros(3)
    w = 2.0 * np.pi * kin.flap_frequency.value
    v_peak = kin.flap_amplitude.value * w * el.s
    q = 0.5 * rho * v_peak**2 * el.chord * (el.area / el.chord)
    return spec.gain * q * window * np.asarray(e_L, dtype=float)


def _wing_tip_W(wing, kin, t: float, R_wb: np.ndarray | None = None,
                p_B_W: np.ndarray | None = None) -> np.ndarray:
    """Tip mid-chord point in world; R_wb/p_B_W give the body pose (identity /
    origin when omitted — fixed-body case)."""
    ks = wing_state(kin, t)
    R_WR, _, _, _ = wing_chain(ks, wing.side)
    R_world_SP = R_B_SP_wing(kin.stroke_plane_angle.value, dihedral_of(wing), wing.side)
    if R_wb is not None:
        R_world_SP = R_wb @ R_world_SP
    x_tip = (wing.pitch_axis_fraction.value - 0.5) * wing.tip_chord.value
    tip_WR = np.array([x_tip, wing.span.value, 0.0])
    hinge_W = np.asarray(wing.hinge_location.value, float)
    if R_wb is not None:
        hinge_W = np.asarray(p_B_W, dtype=float) + R_wb @ hinge_W
    return hinge_W + R_world_SP @ (R_WR @ tip_WR)


def clap_fling_force(wing, el, kin, t: float, e_L: np.ndarray, rho: float,
                     spec, mirror_tip_W: np.ndarray | None = None,
                     own_tip_W: np.ndarray | None = None) -> np.ndarray:
    """Clap-and-fling circulation boost; zero outside the gap threshold.

    mirror_tip_W / own_tip_W: tip positions of the pair (computed by the
    caller, which knows both wings); force scales with the element's local
    dynamic pressure and the gap factor (1 - gap/g_thresh)+.
    """
    g_thresh = spec.params.get("gap_threshold", 0.02)
    gap = float(np.linalg.norm(own_tip_W - mirror_tip_W))
    factor = max(0.0, 1.0 - gap / g_thresh)
    if factor <= 0.0:
        return np.zeros(3)
    v = el.v_W if hasattr(el, "v_W") else np.zeros(3)
    speed = float(np.linalg.norm(v))
    if speed < 1e-14:
        return np.zeros(3)
    q = 0.5 * rho * speed**2 * el.chord * (el.area / el.chord)
    return spec.gain * q * factor * np.asarray(e_L, dtype=float)
