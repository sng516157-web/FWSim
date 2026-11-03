"""§11.2 left/right symmetry tests.

`test_mirrored_wings_mirrored_forces` is real (M1). The zero net roll/yaw
moment test needs the two-wing vehicle assembly and lands in M2.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.environment import still_air
from fwmav_sim.kinematics.blade_element import discretize, element_state
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import AeroModelSpec, Vector3Parameter, WingSpec

_PROV_EST = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}
_PROV_LIT = {"source": "literature", "reference": "Dickinson et al. 1999, Science 284:1954"}
_COEFFS = {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV_EST}


def _wing(side: str) -> WingSpec:
    hinge = [0.01, 0.02, 0.0] if side == "right" else [0.01, -0.02, 0.0]
    return WingSpec(
        name=f"wing_{side}", side=side,
        span=_param(0.3), root_chord=_param(0.08), tip_chord=_param(0.04),
        planform="trapezoidal",
        hinge_location=Vector3Parameter(value=hinge, unit="m", provenance=_PROV_EST),
        pitch_axis_fraction=_param(0.25, "1"),
    )


def _kin() -> PrescribedWingKinematics:
    return PrescribedWingKinematics(
        flap_amplitude=_param(1.05, "rad"), flap_frequency=_param(13.7, "Hz"),
        pitch_amplitude=_param(0.6, "rad"), pitch_phase=_param(0.9, "rad"),
        pitch_offset=_param(0.12, "rad"), deviation_amplitude=_param(0.1, "rad"),
        deviation_phase=_param(0.4, "rad"), stroke_plane_angle=_param(0.2, "rad"),
    )


def test_mirrored_wings_mirrored_forces():
    """frames.md §4: same commands on mirrored wings give F_L = S F_R for
    every force component, at every time and every element."""
    model = QuasiSteadyBladeElement(
        AeroModelSpec(
            name="quasi_steady_blade_element",
            coefficients={k: {"value": v, "unit": "1", "provenance": _PROV_LIT}
                          for k, v in _COEFFS.items()},
        )
    )
    env = still_air()
    S = np.diag([1.0, -1.0, 1.0])
    wing_r, wing_l = _wing("right"), _wing("left")
    els_r, els_l = discretize(wing_r, 8), discretize(wing_l, 8)
    kin = _kin()
    for t in np.linspace(0.0, 1.0 / 13.7, 9):
        for er, el in zip(els_r, els_l, strict=True):
            fr = model.element_forces(element_state(wing_r, er, kin, t, nu=env.nu), env)
            fl = model.element_forces(element_state(wing_l, el, kin, t, nu=env.nu), env)
            np.testing.assert_allclose(fl.F_trans, S @ fr.F_trans, atol=1e-12)
            np.testing.assert_allclose(fl.F_rot, S @ fr.F_rot, atol=1e-12)
            np.testing.assert_allclose(fl.F_am, S @ fr.F_am, atol=1e-12)
            np.testing.assert_allclose(fl.resultant, S @ fr.resultant, atol=1e-12)


@pytest.mark.skip(reason="stub — implemented in M2 (two-wing vehicle assembly)")
def test_symmetric_flapping_zero_roll_yaw_moment():
    raise NotImplementedError
