"""Level-3 abdomen aerodynamics tests (brief §6.5; post-M5 item 4).

- force law vs hand-computed flat-plate values (Dickinson-form fits)
- zero-speed and 90-degree degenerate branches
- plant integration: the abdomen joint reaction torque includes the aero
  moment about the joint axis (frame/placement check)
- end-to-end: short free-body run with abdomen aero active
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.abdomen import AbdomenPlateAero
from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.vehicle.schema import AbdomenAeroSpec, AeroModelSpec, VehicleDefinition

_PROV = {"source": "estimated"}

_COEFS = {k: {"value": v, "unit": "1", "provenance": dict(_PROV)} for k, v in {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.2,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}.items()}

_ENV = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.zeros(3))


def _model(area=2.4e-4, cp=0.02):
    return AbdomenPlateAero(AbdomenAeroSpec.model_validate({
        "area": {"value": area, "unit": "m^2", "provenance": dict(_PROV)},
        "cp_offset": {"value": cp, "unit": "m", "provenance": dict(_PROV)},
        "coefficients": _COEFS,
    }))


def _cl_cd(alpha):
    return QuasiSteadyBladeElement(
        AeroModelSpec(name="quasi_steady_blade_element", coefficients={
            k: dict(v) for k, v in _COEFS.items()})).cl_cd(alpha)


def test_wind_along_axis_pure_drag_plus_fit_lift():
    m = _model()
    U = 3.0
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.array([U, 0.0, 0.0]))
    F, alpha = m.force(np.zeros(3), np.eye(3), env)
    # flow along the abdomen axis: alpha = 0, e_L = -z, v_hat = -x
    assert alpha == pytest.approx(0.0, abs=1e-12)
    cl, cd = _cl_cd(0.0)
    q = 0.5 * 1.225 * U**2 * 2.4e-4
    np.testing.assert_allclose(F, [q * cd, 0.0, -q * cl], atol=1e-15)


def test_zero_speed_zero_force():
    m = _model()
    F, alpha = m.force(np.zeros(3), np.eye(3), _ENV)
    assert np.all(F == 0.0) and alpha == 0.0


def test_normal_flow_drag_only():
    m = _model()
    U = 2.0
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.array([0.0, 0.0, U]))
    F, alpha = m.force(np.zeros(3), np.eye(3), env)
    # flow along the plate normal: lift direction undefined -> pure drag
    _, cd = _cl_cd(alpha)
    q = 0.5 * 1.225 * U**2 * 2.4e-4
    np.testing.assert_allclose(F, [0.0, 0.0, q * cd], atol=1e-15)
    assert abs(abs(alpha) - np.pi / 2) < 1e-12


def test_spanwise_flow_excluded():
    """Flow along y_abd (roll axis) does not enter alpha or the force."""
    m = _model()
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.array([0.0, 5.0, 0.0]))
    F, alpha = m.force(np.zeros(3), np.eye(3), env)
    assert np.all(F == 0.0)  # only spanwise flow -> no plate force


# --- plant integration ----------------------------------------------------------

def _abdomen_vehicle(with_aero=True):
    d = {
        "name": "abd_aero_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": 0.011, "unit": "kg", "provenance": dict(_PROV)},
             "com_offset": {"value": [0, 0, 0], "unit": "m", "provenance": dict(_PROV)},
             "inertia": {"value": (np.eye(3) * 2e-6).tolist(), "unit": "kg m^2",
                         "provenance": dict(_PROV)}},
            {"name": "abdomen",
             "mass": {"value": 0.004, "unit": "kg", "provenance": dict(_PROV)},
             "com_offset": {"value": [-0.015, 0, 0], "unit": "m", "provenance": dict(_PROV)},
             "inertia": {"value": (np.eye(3) * 5e-7).tolist(), "unit": "kg m^2",
                         "provenance": dict(_PROV)}},
        ],
        "joints": [{
            "name": "abdomen_pitch", "type": "revolute",
            "parent_body": "thorax", "child_body": "abdomen",
            "axis": [0.0, 1.0, 0.0],
            "location": {"value": [-0.015, 0, 0], "unit": "m", "provenance": dict(_PROV)},
        }],
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    }
    if with_aero:
        d["abdomen_aero"] = {
            "area": {"value": 2.4e-4, "unit": "m^2", "provenance": dict(_PROV)},
            "cp_offset": {"value": 0.02, "unit": "m", "provenance": dict(_PROV)},
            "coefficients": _COEFS,
        }
    return VehicleDefinition.model_validate(d)


def _plant(with_aero, mode="prescribed"):
    from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
    from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
    from fwmav_sim.simulation.free_body import AbdomenControl, FreeBodyPlant, IsolationSpec

    kin = PrescribedWingKinematics.model_validate({
        "flap_amplitude": {"value": 0.5, "unit": "rad", "provenance": dict(_PROV)},
        "flap_frequency": {"value": 10.0, "unit": "Hz", "provenance": dict(_PROV)},
        "pitch_amplitude": {"value": 0.3, "unit": "rad", "provenance": dict(_PROV)},
        "pitch_phase": {"value": 0.0, "unit": "rad", "provenance": dict(_PROV)},
        "pitch_offset": {"value": 0.0, "unit": "rad", "provenance": dict(_PROV)},
        "stroke_plane_angle": {"value": 0.0, "unit": "rad", "provenance": dict(_PROV)},
    })
    control = AbdomenControl(
        mode=mode,
        profile=AbdomenProfile(kind="sinusoid", amplitude=0.3, frequency=3.0)
        if mode == "prescribed" else None)
    return FreeBodyPlant(_abdomen_vehicle(with_aero), kin, _ENV, None,
                         control=control,
                         isolation=IsolationSpec(abdomen_aero_enabled=with_aero))


def _y0():
    from fwmav_sim.simulation.integrator import pack_state
    return pack_state([0, 0, 0], [1, 0, 0, 0], [0.0], [0, 0, 0], [0, 0, 0], [0.0])


def test_abdomen_force_placement_matches_prescribed_kinematics():
    """The plant's recorded abdomen aero force must equal the force-law
    evaluation at the CoP velocity/orientation computed independently from the
    prescribed profile (frame and placement check, dynamics-free)."""
    from fwmav_sim.core.transforms import Ry

    t = 0.09  # alpha_dot != 0 for the 3 Hz sinusoid
    plant = _plant(True)
    plant.rhs(t, _y0())
    F = plant.last_F_abdomen_aero
    assert F is not None and np.linalg.norm(F) > 0

    # independent kinematics: alpha = 0.3 sin(2 pi 3 t), pitch about +y
    w = 2 * np.pi * 3.0
    alpha_t = 0.3 * np.sin(w * t)
    ad_t = 0.3 * w * np.cos(w * t)
    R_abd = Ry(alpha_t)
    r_cop_W = np.array([-0.015, 0.0, 0.0]) + R_abd @ np.array([-0.02, 0, 0])
    v_cop_W = np.cross(np.array([0.0, ad_t, 0.0]), r_cop_W - np.array([-0.015, 0, 0]))
    F_expect, _ = _model().force(v_cop_W, R_abd, _ENV)
    np.testing.assert_allclose(F, F_expect, rtol=1e-12)


def test_reaction_torque_changes_with_abdomen_aero():
    """The recorded actuator reaction torque must respond when abdomen aero is
    active (sign/opposite of the aero load); the exact value depends on the
    base acceleration, so this pins presence and sign, not magnitude."""
    t = 0.09
    on = _plant(True)
    off = _plant(False)
    on.rhs(t, _y0())
    off.rhs(t, _y0())
    delta = on.last_reaction[0] - off.last_reaction[0]
    F = on.last_F_abdomen_aero
    from fwmav_sim.core.transforms import Ry
    alpha_t = 0.3 * np.sin(2 * np.pi * 3.0 * t)
    arm = Ry(alpha_t) @ np.array([-0.02, 0.0, 0.0])
    moment_y = np.cross(arm, F)[1]
    assert abs(delta) > 0
    assert np.sign(delta) == np.sign(-moment_y)


def test_free_body_run_with_abdomen_aero():
    from fwmav_sim.simulation.runner import run_free_body
    from fwmav_sim.vehicle.schema import SolverSpec

    plant = _plant(True)
    ts, status, _ = run_free_body(plant, SolverSpec(max_time=0.05), _y0())
    assert status == "OK"
    F = ts["abdomen/F_aero"]
    alpha = ts["abdomen/alpha_aero"]
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(alpha))
    assert np.linalg.norm(F, axis=1).max() > 0.0  # the swinging abdomen sees air
