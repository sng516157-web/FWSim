"""§11.1 individual aerodynamic term tests (M1 Task 5).

Hand-computed references against the cited formulas (Dickinson et al. 1999;
Sane & Dickinson 2002; flat-plate added mass). Element states are constructed
directly so each term is tested in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.environment import still_air
from fwmav_sim.kinematics.blade_element import ElementState
from fwmav_sim.vehicle.schema import AeroModelSpec

# Dickinson et al. 1999 Robofly fit coefficients (provenance: literature)
_COEFFS = {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}
_PROV = {"source": "literature", "reference": "Dickinson et al. 1999, Science 284:1954"}
_RHO = 1.225
_CHORD = 0.08
_AREA = 0.008  # dr = 0.1


def _model() -> QuasiSteadyBladeElement:
    return QuasiSteadyBladeElement(
        AeroModelSpec(
            name="quasi_steady_blade_element",
            coefficients={k: {"value": v, "unit": "1", "provenance": _PROV}
                          for k, v in _COEFFS.items()},
        )
    )


def _state(v_W, a_W=(0.0, 0.0, 0.0), alpha=0.0, pitch_rate=0.0, side="right") -> ElementState:
    return ElementState(
        r_W=np.zeros(3),
        v_W=np.asarray(v_W, dtype=float),
        a_W=np.asarray(a_W, dtype=float),
        R_W_BE=np.eye(3),
        omega_W=np.zeros(3),
        chord=_CHORD,
        area=_AREA,
        alpha=alpha,
        reynolds=0.0,
        side=side,
        pitch_rate=pitch_rate,
        pitch_axis_fraction=0.5,
    )


_Q = 0.5 * _RHO * 1.0**2 * _CHORD * 0.1  # dynamic pressure term at |v| = 1


def test_translational_term_at_0_45_90deg():
    m = _model()
    env = still_air(rho=_RHO)

    # alpha = 0: element moving chord-first (+x)
    cl0 = 0.225 + 1.58 * np.sin(np.radians(-7.20))
    cd0 = 1.92 - 1.55 * np.cos(np.radians(-9.82))
    f = m.element_forces(_state([1.0, 0.0, 0.0], alpha=0.0), env)
    np.testing.assert_allclose(
        f.F_trans, _Q * (cl0 * np.array([0, 0, -1]) - cd0 * np.array([1, 0, 0])), rtol=1e-12
    )
    assert f.F_trans[2] < 0  # lift points UP (-z, frames.md section 1.1)
    assert f.F_trans[0] < 0  # drag opposes +x motion

    # alpha = +45 deg: element moving forward and DOWN (leading edge up
    # relative to the oncoming flow; flow approaches from the underside)
    a = np.radians(45.0)
    v = np.array([np.cos(a), 0.0, np.sin(a)])
    cl45 = 0.225 + 1.58 * np.sin(np.radians(2.13 * 45 - 7.20))
    cd45 = 1.92 - 1.55 * np.cos(np.radians(2.04 * 45 - 9.82))
    n_perp = np.array([0.0, 0.0, 1.0]) - np.sin(a) * v
    e_L = -n_perp / np.linalg.norm(n_perp)
    f = m.element_forces(_state(v, alpha=a), env)
    np.testing.assert_allclose(f.F_trans, _Q * (cl45 * e_L - cd45 * v), rtol=1e-12)

    # alpha = 90 deg: normal flow, lift direction undefined -> drag only
    cd90 = 1.92 - 1.55 * np.cos(np.radians(2.04 * 90 - 9.82))
    f = m.element_forces(_state([0.0, 0.0, -1.0], alpha=np.pi / 2), env)
    np.testing.assert_allclose(f.F_trans, _Q * (-cd90 * np.array([0.0, 0.0, -1.0])), rtol=1e-12)


def test_translational_negative_alpha_symmetry():
    """Flat-plate symmetry: C_L(-a) = -C_L(a), C_D(-a) = C_D(a) (docstring)."""
    m = _model()
    for a in (10.0, 45.0, 80.0):
        cl_p, cd_p = m.cl_cd(np.radians(a))
        cl_n, cd_n = m.cl_cd(np.radians(-a))
        assert abs(cl_p + cl_n) < 1e-12 and abs(cd_p - cd_n) < 1e-12


def test_rotational_term_hand_computed():
    """Sane & Dickinson 2002 form: F = rho |v| pi c^2 (3/4 - x0) theta_w_dot dr e_L."""
    m = _model()
    env = still_air(rho=_RHO)
    st = _state([1.0, 0.0, 0.0], alpha=0.0, pitch_rate=10.0)
    f = m.element_forces(st, env)
    gamma = np.pi * _CHORD**2 * 10.0 * (0.75 - 0.5)
    expected = _RHO * 1.0 * gamma * 0.1 * np.array([0.0, 0.0, -1.0])  # e_L at alpha=0
    np.testing.assert_allclose(f.F_rot, expected, rtol=1e-12)
    assert f.F_rot[2] < 0  # pitch-up during forward translation adds lift
    # opposite pitch rate flips the sign
    f_neg = m.element_forces(_state([1.0, 0.0, 0.0], alpha=0.0, pitch_rate=-10.0), env)
    np.testing.assert_allclose(f_neg.F_rot, -expected, rtol=1e-12)


def test_added_mass_term_hand_computed():
    """Flat-plate apparent mass rho pi c^2/4 dr opposing normal acceleration."""
    m = _model()
    env = still_air(rho=_RHO)
    st = _state([0.0, 0.0, 0.0], a_W=[0.0, 0.0, 2.0])
    f = m.element_forces(st, env)
    m_app = _RHO * np.pi * _CHORD**2 / 4.0 * 0.1
    np.testing.assert_allclose(f.F_am, -m_app * 2.0 * np.array([0.0, 0.0, 1.0]), rtol=1e-12)
    assert f.F_am[2] < 0  # reaction opposes the downward (+n_phys) acceleration
    # acceleration along the chord produces no added-mass force
    f2 = m.element_forces(_state([0.0, 0.0, 0.0], a_W=[3.0, 0.0, 0.0]), env)
    np.testing.assert_allclose(f2.F_am, np.zeros(3), atol=1e-18)


def test_zero_relative_velocity_zero_translational_rotational_force():
    """Stage-file test: zero relative velocity => zero force — for the
    velocity-driven terms. Documented exception: added mass responds to
    acceleration even at zero speed."""
    m = _model()
    env = still_air(rho=_RHO)
    f = m.element_forces(_state([0.0, 0.0, 0.0]), env)
    np.testing.assert_allclose(f.F_trans, np.zeros(3), atol=1e-18)
    np.testing.assert_allclose(f.F_rot, np.zeros(3), atol=1e-18)
    np.testing.assert_allclose(f.F_am, np.zeros(3), atol=1e-18)
    f_acc = m.element_forces(_state([0.0, 0.0, 0.0], a_W=[0.0, 0.0, 1.0]), env)
    assert np.linalg.norm(f_acc.F_am) > 0  # documented added-mass exception


def test_drag_opposes_relative_motion():
    m = _model()
    env = still_air(rho=_RHO)
    rng = np.random.default_rng(3)
    for _ in range(50):
        v = rng.normal(size=3)
        alpha = float(np.arctan2(v[2], v[0]))  # signed alpha, same convention as code
        f = m.element_forces(_state(v, alpha=alpha), env)
        assert f.F_trans @ v < 0  # resultant translational force removes energy


def test_missing_coefficients_rejected():
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        QuasiSteadyBladeElement(AeroModelSpec(name="quasi_steady_blade_element", coefficients={}))
