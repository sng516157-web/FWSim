"""§11.1 CoM / parallel-axis inertia / joint-limit tests (M2 Tasks 1-2, 6;
updated in M3 for the serial pitch->roll chain and joint-centred frames).

Hand-computed configurations: (a) symmetric neutral, (b) abdomen deflected
30 deg, (c) point-mass limit. Plus inertia symmetric-PD on random
configurations and joint-limit clamping of the prescribed abdomen profile.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.dynamics.mass_properties import com, inertia_about_com
from fwmav_sim.vehicle.definition import validate_vehicle
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}

_M_T = 0.010
_M_A = 0.004
_D = 0.015  # joint 15 mm behind thorax origin
_L_A = 0.015  # abdomen CoM 15 mm behind the joint (at alpha = 0)
_IT = 2.0e-6
_IA = 5.0e-7


def _vehicle() -> VehicleDefinition:
    v = VehicleDefinition.model_validate({
        "name": "m2_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": _M_T, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[_IT, 0, 0], [0, _IT, 0], [0, 0, _IT]],
                         "unit": "kg m^2", "provenance": _PROV}},
            {"name": "abdomen_roll_carrier",
             "mass": {"value": 0.0, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                         "unit": "kg m^2", "provenance": _PROV},
             "massless": True},
            {"name": "abdomen",
             "mass": {"value": _M_A, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [-_L_A, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[_IA, 0, 0], [0, _IA, 0], [0, 0, _IA]],
                         "unit": "kg m^2", "provenance": _PROV}},
        ],
        "joints": [
            {"name": "abdomen_pitch", "type": "revolute",
             "parent_body": "thorax", "child_body": "abdomen_roll_carrier",
             "axis": [0.0, 1.0, 0.0],
             "location": {"value": [-_D, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "limits": [-0.6, 0.6],
             "damping": {"value": 1e-6, "unit": "N m s / rad", "provenance": _PROV}},
            {"name": "abdomen_roll", "type": "revolute",
             "parent_body": "abdomen_roll_carrier", "child_body": "abdomen",
             "axis": [1.0, 0.0, 0.0],
             "location": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "limits": [-0.6, 0.6]},
        ],
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    })
    validate_vehicle(v)
    return v


def _q(alpha_abd: float = 0.0, beta_abd: float = 0.0, x=0.0, y=0.0, z=0.0,
       psi=0.0, theta=0.0, phi_r=0.0):
    return np.array([x, y, z, psi, theta, phi_r, alpha_abd, beta_abd])


def test_com_hand_computed():
    v = _vehicle()
    # (a) symmetric neutral: abdomen straight back along -x
    expected = np.array([-_M_A * (_D + _L_A) / (_M_T + _M_A), 0.0, 0.0])
    np.testing.assert_allclose(com(_q(0.0), v), expected, atol=1e-15)

    # (b) abdomen deflected 30 deg about +y (tip down, frozen sign rule)
    a = np.deg2rad(30.0)
    r_a = np.array([-_D - _L_A * np.cos(a), 0.0, _L_A * np.sin(a)])
    expected = _M_A / (_M_T + _M_A) * r_a
    np.testing.assert_allclose(com(_q(a), v), expected, atol=1e-15)

    # (b2) roll about the post-pitch x-axis: the abdomen CoM lies ON the roll
    # axis (com_offset = [-L, 0, 0]), so roll leaves the CoM invariant, with
    # or without pitch
    b = np.deg2rad(20.0)
    np.testing.assert_allclose(com(_q(0.0, b), v), com(_q(0.0, 0.0), v), atol=1e-15)
    np.testing.assert_allclose(com(_q(a, b), v), com(_q(a, 0.0), v), atol=1e-15)


def test_inertia_parallel_axis_hand_computed():
    v = _vehicle()
    # (a) neutral: point masses on the x-axis; I about CoM by parallel axis
    r_com = com(_q(0.0), v)
    d_t = -r_com  # thorax at origin
    d_a = np.array([-_D - _L_A, 0.0, 0.0]) - r_com
    I_exp = (_IT + _IA) * np.eye(3)
    for m, d in ((_M_T, d_t), (_M_A, d_a)):
        I_exp += m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    np.testing.assert_allclose(inertia_about_com(_q(0.0), v), I_exp, atol=1e-15)

    # (c) point-mass limit: negligible body inertias
    eps = 1e-12
    v_pm = v.model_copy(deep=True)
    for b in v_pm.bodies:
        if not b.massless:
            b.inertia.value = [[eps, 0, 0], [0, eps, 0], [0, 0, eps]]
    a = np.deg2rad(30.0)
    q = _q(a)
    r_com = com(q, v_pm)
    d_a = np.array([-_D - _L_A * np.cos(a), 0.0, _L_A * np.sin(a)]) - r_com
    d_t = -r_com
    I_exp = _M_T * ((d_t @ d_t) * np.eye(3) - np.outer(d_t, d_t)) \
        + _M_A * ((d_a @ d_a) * np.eye(3) - np.outer(d_a, d_a)) + 2 * eps * np.eye(3)
    np.testing.assert_allclose(inertia_about_com(q, v_pm), I_exp, atol=1e-14)


def test_inertia_symmetric_positive_definite_random_configs():
    v = _vehicle()
    rng = np.random.default_rng(11)
    for _ in range(100):
        q = _q(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
               **{"psi": rng.uniform(-np.pi, np.pi),
                  "theta": rng.uniform(-1.4, 1.4),
                  "phi_r": rng.uniform(-np.pi, np.pi)})
        I_mat = inertia_about_com(q, v)
        np.testing.assert_allclose(I_mat, I_mat.T, atol=1e-18)
        assert np.all(np.linalg.eigvalsh(I_mat) > 0)


def test_joint_limit_clamping():
    """Prescribed abdomen profile clamps to the JointSpec limits."""
    from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha

    prof = AbdomenProfile(kind="sinusoid", amplitude=1.0, frequency=2.0)
    a, _, _ = prescribed_alpha(prof, 0.125, limits=(-0.6, 0.6))  # sin(pi/2)=1 > limit
    assert a == 0.6
    a, _, _ = prescribed_alpha(prof, 0.375, limits=(-0.6, 0.6))
    assert a == -0.6
