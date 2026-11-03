"""§11.2 stationary wing / uniform translation tests (M1)."""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.environment import still_air, uniform_wind
from fwmav_sim.kinematics.blade_element import discretize, element_state
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import AeroModelSpec, WingSpec

_PROV_EST = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}
_PROV_LIT = {"source": "literature", "reference": "Dickinson et al. 1999, Science 284:1954"}
_COEFFS = {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV_EST}


def _wing() -> WingSpec:
    return WingSpec(
        name="wing_right", side="right",
        span=_param(0.3), root_chord=_param(0.08), tip_chord=_param(0.08),
        planform="rectangular",
        hinge_location={"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV_EST},
        pitch_axis_fraction=_param(0.5, "1"),
    )


def _still_kinematics() -> PrescribedWingKinematics:
    return PrescribedWingKinematics(
        flap_amplitude=_param(0.0, "rad"), flap_frequency=_param(10.0, "Hz"),
        pitch_amplitude=_param(0.0, "rad"), pitch_phase=_param(0.0, "rad"),
        pitch_offset=_param(0.0, "rad"), stroke_plane_angle=_param(0.0, "rad"),
    )


def _model() -> QuasiSteadyBladeElement:
    return QuasiSteadyBladeElement(
        AeroModelSpec(
            name="quasi_steady_blade_element",
            coefficients={k: {"value": v, "unit": "1", "provenance": _PROV_LIT}
                          for k, v in _COEFFS.items()},
        )
    )


def test_stationary_wing_zero_force():
    """Stationary wing + still air => every force term exactly zero."""
    model, wing, kin = _model(), _wing(), _still_kinematics()
    env = still_air()
    for t in (0.0, 0.013, 0.05):
        for el in discretize(wing, 8):
            st = element_state(wing, el, kin, t, nu=env.nu)
            f = model.element_forces(st, env)
            np.testing.assert_allclose(st.v_W, np.zeros(3), atol=1e-15)
            np.testing.assert_allclose(st.a_W, np.zeros(3), atol=1e-12)
            for term in (f.F_trans, f.F_rot, f.F_am, f.resultant):
                np.testing.assert_allclose(term, np.zeros(3), atol=1e-15)


def test_uniform_translation_steady_drag_sign():
    """Uniform flow from ahead (-x wind, wing at rest, alpha = 0):
    steady force = q (C_L(0) e_L - C_D(0) v_hat), drag pushing the wing -x."""
    model, wing, kin = _model(), _wing(), _still_kinematics()
    U = 2.0
    env = uniform_wind(np.array([-U, 0.0, 0.0]))
    cl0, cd0 = model.cl_cd(0.0)
    for el in discretize(wing, 8):
        st = element_state(wing, el, kin, 0.0, nu=env.nu, v_air_W=env.v_air_W)
        assert abs(st.alpha) < 1e-12
        f = model.element_forces(st, env)
        q = 0.5 * env.rho * U**2 * el.chord * (el.area / el.chord)
        np.testing.assert_allclose(
            f.F_trans, q * (cl0 * np.array([0, 0, -1]) - cd0 * np.array([1, 0, 0])), rtol=1e-12
        )
        np.testing.assert_allclose(f.F_rot, np.zeros(3), atol=1e-18)
        np.testing.assert_allclose(f.F_am, np.zeros(3), atol=1e-18)
        assert f.F_trans[0] < 0  # drag pushes downstream (-x)
        assert f.F_trans[2] < 0  # small positive C_L(0) lifts up (-z)
