"""§11.1 kinematic derivative tests (M1 Task 2)."""

from __future__ import annotations

import numpy as np

from fwmav_sim.kinematics.prescribed import (
    FourierTerm,
    PrescribedWingKinematics,
    wing_state,
)

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}


def _param(value: float, unit: str = "rad") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV}


def _kin(**overrides) -> PrescribedWingKinematics:
    fields = {
        "flap_amplitude": _param(1.05),
        "flap_frequency": _param(13.7, "Hz"),
        "pitch_amplitude": _param(0.6),
        "pitch_phase": _param(0.9),
        "pitch_offset": _param(0.1),
        "deviation_amplitude": _param(0.15),
        "deviation_phase": _param(0.3),
        "stroke_plane_angle": _param(0.2),
        "pitch_fourier": [FourierTerm(amplitude=0.08, harmonic=2, phase=0.5)],
    }
    fields.update(overrides)
    return PrescribedWingKinematics(**fields)


def test_prescribed_kinematics_matches_finite_difference():
    rng = np.random.default_rng(7)
    kin = _kin()
    for t in rng.uniform(0.0, 1.0, 100):
        for name in ("phi", "theta_w", "psi_w"):
            s0 = wing_state(kin, t)
            h1 = 1e-6  # first derivative: truncation ~h^2 f''' , roundoff ~eps/h
            sm, sp = wing_state(kin, t - h1), wing_state(kin, t + h1)
            d_fd = (getattr(sp, name) - getattr(sm, name)) / (2 * h1)
            h2 = 1e-5  # second derivative: trunc ~h^2 f'''' ~4e-4 here; roundoff ~eps*(w t)/h^2
            sm, sp = wing_state(kin, t - h2), wing_state(kin, t + h2)
            s0 = wing_state(kin, t)
            dd_fd = (getattr(sp, name) - 2 * getattr(s0, name) + getattr(sm, name)) / h2**2
            assert abs(getattr(s0, name + "_dot") - d_fd) < 1e-6 * (1 + abs(d_fd)), (name, t)
            # looser bound for ddot: central-difference noise floor with large phase w*t
            assert abs(getattr(s0, name + "_ddot") - dd_fd) < 1e-5 * (1 + abs(dd_fd)), (name, t)


def test_reversed_phi_amp_documented_sign_change():
    """frames.md §3: negating phi_amp negates the stroke angle and its
    derivatives, and leaves theta_w / psi_w untouched."""
    kin = _kin()
    kin_rev = _kin(flap_amplitude=_param(-1.05))
    for t in np.linspace(0.0, 0.1, 21):
        a, b = wing_state(kin, t), wing_state(kin_rev, t)
        assert a.phi == -b.phi
        assert a.phi_dot == -b.phi_dot
        assert a.phi_ddot == -b.phi_ddot
        assert a.theta_w == b.theta_w
        assert a.psi_w == b.psi_w


def test_zero_deviation_when_unset():
    kin = _kin(deviation_amplitude=None, deviation_phase=None)
    s = wing_state(kin, 0.03)
    assert s.psi_w == 0.0 and s.psi_w_dot == 0.0 and s.psi_w_ddot == 0.0
