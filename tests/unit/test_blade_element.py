"""§11.1 blade-element velocity/angle tests (M1 Task 3).

Hand-computed reference configuration: beta = 0, right wing, rectangular
planform, mid-chord pitch axis, theta_w = psi_w = 0, evaluated at t = T/4
(mid-backstroke): phi = 0, phi_dot = -phi_amp * w, phi_ddot = 0.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.kinematics.blade_element import (
    discretize,
    element_state,
    planform_area,
)
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import WingSpec

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV}


def _wing(side: str = "right", pitch_axis: float = 0.5, planform: str = "rectangular") -> WingSpec:
    return WingSpec(
        name=f"wing_{side}",
        side=side,
        span=_param(0.3),
        root_chord=_param(0.08),
        tip_chord=_param(0.04),
        planform=planform,
        hinge_location={"value": [0.01, 0.02, 0.0], "unit": "m", "provenance": _PROV},
        pitch_axis_fraction=_param(pitch_axis, "1"),
    )


def _kin(phi_amp: float = 1.0, f: float = 10.0) -> PrescribedWingKinematics:
    return PrescribedWingKinematics(
        flap_amplitude=_param(phi_amp, "rad"),
        flap_frequency=_param(f, "Hz"),
        pitch_amplitude=_param(0.0, "rad"),
        pitch_phase=_param(0.0, "rad"),
        pitch_offset=_param(0.0, "rad"),
        stroke_plane_angle=_param(0.0, "rad"),
    )


_NU = 1.5e-5  # m^2/s


def test_element_state_hand_computed_mid_backstroke():
    wing = _wing()
    kin = _kin()
    elements = discretize(wing, 4)
    el = elements[1]  # s = 0.1125 m, chord 0.08 m, x_c = 0 (mid-chord axis)
    T = 1.0 / kin.flap_frequency.value
    w = 2 * np.pi * kin.flap_frequency.value
    phi_amp = kin.flap_amplitude.value
    st = element_state(wing, el, kin, T / 4, nu=_NU)

    # position: hinge + [0, s, 0] (identity chain at phi = theta_w = psi_w = 0)
    np.testing.assert_allclose(st.r_W, [0.01, 0.02 + el.s, 0.0], atol=1e-12)
    # velocity: v = omega x r with omega = [0, 0, -phi_dot] -> v = [phi_dot * s, 0, 0]
    np.testing.assert_allclose(st.v_W, [-phi_amp * w * el.s, 0.0, 0.0], atol=1e-12)
    # acceleration: centripetal toward the root, -phi_dot^2 * s along y
    np.testing.assert_allclose(st.a_W, [0.0, -(phi_amp * w) ** 2 * el.s, 0.0], atol=1e-9)
    # orientation: identity
    np.testing.assert_allclose(st.R_W_BE, np.eye(3), atol=1e-12)
    # AoA: wing moving in -x (trailing edge first) -> alpha = 180 deg
    assert abs(abs(st.alpha) - np.pi) < 1e-12
    # Reynolds number
    assert st.reynolds == abs(-phi_amp * w * el.s) * el.chord / _NU


def test_element_velocity_hand_computed():
    """Forward extreme (t = 0): phi = phi_amp, phi_dot = 0 -> element at rest
    instantaneously; with nonzero pitch rate the velocity is rotational."""
    wing = _wing()
    kin = PrescribedWingKinematics(
        flap_amplitude=_param(1.0, "rad"),
        flap_frequency=_param(10.0, "Hz"),
        pitch_amplitude=_param(0.5, "rad"),
        pitch_phase=_param(0.0, "rad"),
        pitch_offset=_param(0.0, "rad"),
        stroke_plane_angle=_param(0.0, "rad"),
    )
    el = discretize(wing, 4)[1]
    st = element_state(wing, el, kin, 0.0, nu=_NU)
    # at t=0: phi=1, phi_dot=0, theta_w=0, theta_w_dot = 0.5*w; R = Rz(-1);
    # omega_SP = e_y * theta_w_dot -> v = theta_w_dot * (y_WR x r)
    w = 2 * np.pi * 10.0
    R = np.array(
        [[np.cos(-1.0), -np.sin(-1.0), 0.0],
         [np.sin(-1.0), np.cos(-1.0), 0.0],
         [0.0, 0.0, 1.0]]
    )
    r = R @ np.array([0.0, el.s, 0.0])
    v_expected = np.cross(R @ np.array([0.0, 0.5 * w, 0.0]), r)
    np.testing.assert_allclose(st.v_W, v_expected, atol=1e-10)
    np.testing.assert_allclose(st.r_W, [0.01, 0.02, 0.0] + r, atol=1e-12)


def test_element_aoa_hand_computed():
    """Pure prescribed translation cases pin the alpha sign convention."""
    wing = _wing()
    kin = _kin(phi_amp=0.0)  # wing at rest at phi = 0: v_rel = -v_air exactly
    el = discretize(wing, 1)[0]

    def alpha_for(v_air):
        return element_state(wing, el, kin, 0.0, nu=_NU, v_air_W=v_air).alpha

    # wind from ahead (-x): oncoming flow along +x hits the leading edge -> alpha = 0
    assert abs(alpha_for(np.array([-1.0, 0.0, 0.0]))) < 1e-12
    # wind from above (-z) -> v_rel = +z (toward physical normal) -> alpha = -90 deg
    assert abs(alpha_for(np.array([0.0, 0.0, -1.0])) + np.pi / 2) < 1e-12
    # wind from below -> alpha = +90 deg
    assert abs(alpha_for(np.array([0.0, 0.0, 1.0])) - np.pi / 2) < 1e-12
    # wind at 45 deg from front-below
    assert abs(alpha_for(np.array([-1.0, 0.0, 1.0])) - np.pi / 4) < 1e-12


def test_element_areas_sum_to_planform_area():
    for planform, expected in [
        ("rectangular", 0.3 * 0.08),
        ("trapezoidal", 0.3 * (0.08 + 0.04) / 2),
        ("elliptical", np.pi * 0.3 * 0.08 / 4),
    ]:
        wing = _wing(planform=planform)
        assert abs(planform_area(wing) - expected) < 1e-12
        for method in ("equal_span", "equal_area"):
            for n in (1, 4, 16, 64):
                total = sum(el.area for el in discretize(wing, n, method))
                assert abs(total - expected) < 1e-10, (planform, method, n)


def test_equal_area_discretization_equal_areas():
    wing = _wing(planform="elliptical")
    elements = discretize(wing, 8, "equal_area")
    areas = [el.area for el in elements]
    np.testing.assert_allclose(areas, areas[0], rtol=1e-10)


def test_mirrored_wings_mirrored_states():
    """frames.md §4: left element position/velocity/acceleration mirror the
    right under S = diag(1,-1,1); alpha is identical (physical-normal rule)."""
    kin = _kin()
    kin_w = PrescribedWingKinematics(
        **{**kin.model_dump(), "pitch_amplitude": _param(0.4, "rad"),
           "pitch_phase": _param(0.7, "rad"), "pitch_offset": _param(0.15, "rad"),
           "stroke_plane_angle": _param(0.25, "rad")}
    )
    wing_r = _wing("right", pitch_axis=0.0)
    # the left wing's hinge is the physical mirror of the right's (frames.md §4)
    from fwmav_sim.vehicle.schema import Vector3Parameter

    wing_l = _wing("left", pitch_axis=0.0).model_copy(
        update={"hinge_location": Vector3Parameter(value=[0.01, -0.02, 0.0], unit="m",
                                                   provenance=_PROV)}
    )
    els_r = discretize(wing_r, 8)
    els_l = discretize(wing_l, 8)
    S = np.diag([1.0, -1.0, 1.0])
    for t in np.linspace(0, 0.05, 6):
        for er, el in zip(els_r, els_l, strict=True):
            sr = element_state(wing_r, er, kin_w, t, nu=_NU)
            sl = element_state(wing_l, el, kin_w, t, nu=_NU)
            np.testing.assert_allclose(sl.r_W, S @ sr.r_W, atol=1e-12)
            np.testing.assert_allclose(sl.v_W, S @ sr.v_W, atol=1e-12)
            np.testing.assert_allclose(sl.a_W, S @ sr.a_W, atol=1e-10)
            assert abs(sl.alpha - sr.alpha) < 1e-12
            assert sl.chord == sr.chord and sl.area == sr.area
