"""M3.0 spike + §11.1 unit tests for the spatial-vector core (dynamics/spatial.py).

Validations (per M3 stage file Task 1):
- spatial transforms vs their physical meaning / 4x4 homogeneous adjoint
- spatial inertia kinetic-energy identity
- CRBA symmetry + positive definiteness on random configurations
- **planar restriction of CRBA/RNEA vs the M2 SymPy-derived reference** (the
  trust-anchor check)
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import R_W_B, quat_from_euler
from fwmav_sim.dynamics import _reduced_reference as ref
from fwmav_sim.dynamics.spatial import (
    ArticulatedModel,
    spatial_inertia,
    xform_force,
    xform_motion,
)
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}

# geometry matching the M2 reduced-model parameters
_M_T, _M_A = 0.011, 0.004
_IT, _IA = 2.0e-6, 5.0e-7
_D, _L_A = 0.015, 0.015


def _vehicle() -> VehicleDefinition:
    return VehicleDefinition.model_validate({
        "name": "m3_spatial_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": _M_T, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[_IT, 0, 0], [0, _IT, 0], [0, 0, _IT]],
                         "unit": "kg m^2", "provenance": _PROV}},
            {"name": "abdomen_roll_carrier",
             "mass": {"value": 0.0, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[0.0] * 3] * 3, "unit": "kg m^2", "provenance": _PROV},
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
             "location": {"value": [-_D, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
            {"name": "abdomen_roll", "type": "revolute",
             "parent_body": "abdomen_roll_carrier", "child_body": "abdomen",
             "axis": [1.0, 0.0, 0.0],
             "location": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
        ],
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    })


def test_xform_motion_matches_physical_meaning():
    """xform_motion(R, p): p = origin of the TARGET frame A in source coords.
    R = I, p = [1,0,0], body spinning about its own origin (B): the material
    point at A's origin sits at +p from B's origin, so v_A = omega x p."""
    X = xform_motion(np.eye(3), np.array([1.0, 0.0, 0.0]))
    V_B = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])  # omega about y only
    V_A = X @ V_B
    np.testing.assert_allclose(V_A[:3], V_B[:3], atol=1e-15)
    np.testing.assert_allclose(V_A[3:], np.cross(V_B[:3], [1.0, 0.0, 0.0]), atol=1e-15)


def test_xform_dual_relationship():
    rng = np.random.default_rng(0)
    for _ in range(50):
        psi, theta, phi_r = rng.uniform(-1.0, 1.0, 3)
        R = R_W_B(psi, theta, phi_r)
        p = rng.normal(size=3)
        X_m = xform_motion(R, p)
        X_f = xform_force(R, p)
        np.testing.assert_allclose(X_f, np.linalg.inv(X_m).T, atol=1e-12)


def test_spatial_inertia_kinetic_energy_identity():
    """T = 1/2 V' I_spatial V == 1/2 m |v_com|^2 + 1/2 omega' I_c omega."""
    rng = np.random.default_rng(1)
    m = 0.013
    c = np.array([-0.02, 0.005, 0.003])
    I_c = np.diag([2e-6, 3e-6, 1.5e-6])
    I_sp = spatial_inertia(m, c, I_c)
    for _ in range(50):
        V = rng.normal(size=6)
        omega, v_O = V[:3], V[3:]
        v_com = v_O + np.cross(omega, c)
        T_expected = 0.5 * m * float(v_com @ v_com) + 0.5 * float(omega @ I_c @ omega)
        assert abs(0.5 * float(V @ I_sp @ V) - T_expected) < 1e-15


def test_crba_symmetric_positive_definite():
    model = ArticulatedModel(_vehicle())
    rng = np.random.default_rng(2)
    for _ in range(100):
        q = np.concatenate([
            rng.normal(size=3) * 0.1,
            quat_from_euler(*rng.uniform(-1.0, 1.0, 3)),
            rng.uniform(-0.6, 0.6, 2),
        ])
        M = model.crba(q)
        np.testing.assert_allclose(M, M.T, atol=1e-15)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def _planar_map(theta: float) -> np.ndarray:
    """A (8x4) mapping M2 accelerations [xdd, zdd, thdd, add] onto the M3
    ordering [omega_B_dot; v_B_dot; alpha_dd; beta_dd] (v at rest: v_B = R' v_W)."""
    c, s = np.cos(theta), np.sin(theta)
    A = np.zeros((8, 4))
    A[1, 2] = 1.0  # omega_y = theta_dot
    A[3, 0] = c   # v_Bx
    A[3, 2] = 0.0
    A[5, 0] = s   # v_Bz
    A[3, 1] = -s
    A[5, 1] = c
    A[6, 3] = 1.0  # alpha
    return A


def test_crba_planar_restriction_matches_sympy():
    """The M3.0 spike criterion: M_planar(q) == M2 SymPy free_M to < 1e-10."""
    model = ArticulatedModel(_vehicle())
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(100):
        theta, alpha = rng.uniform(-1.0, 1.0, 2)
        q = np.concatenate([[0.0, 0.0, 0.0], quat_from_euler(0.0, theta, 0.0),
                            [alpha, 0.0]])
        A = _planar_map(theta)
        M3_planar = A.T @ model.crba(q) @ A
        M2 = ref.free_M(theta, alpha, _M_T, _M_A, _IT, _IA, _D, _L_A)
        worst = max(worst, float(np.abs(M3_planar - M2).max()))
        np.testing.assert_allclose(M3_planar, M2, atol=1e-10)
    print(f"\nplanar CRBA vs SymPy: max abs diff = {worst:.2e}")


def test_rnea_bias_planar_matches_sympy():
    """bias = C u + G: planar restriction vs M2 bias_vector + gravity_vector.

    Coordinate note: the M2 reduced model uses world-frame velocities
    [xd, zd, theta_d, alpha_d] while the spatial model uses body-frame
    quasi-velocities [omega_B; v_B; ...] with v_B = R_W_B(theta)^T v_W.
    Since the velocity map A(theta) is configuration-dependent, the
    generalized biases relate as bias2 = A^T bias3 + A^T M3 (dA/dtheta) v2
    (the Coriolis terms of the coordinate change itself)."""
    model = ArticulatedModel(_vehicle())
    rng = np.random.default_rng(4)
    for _ in range(50):
        theta, alpha = rng.uniform(-1.0, 1.0, 2)
        xd, zd, thd, ald = rng.uniform(-2.0, 2.0, 4)
        q = np.concatenate([[0.0, 0.0, 0.0], quat_from_euler(0.0, theta, 0.0),
                            [alpha, 0.0]])
        c, s = np.cos(theta), np.sin(theta)
        # u3 = [omega_B; v_B; alpha_dot; beta_dot] with v_B = R' v_W
        u = np.array([0.0, thd, 0.0, c * xd - s * zd, 0.0, s * xd + c * zd, ald, 0.0])
        bias3 = model.bias(q, u)
        A = _planar_map(theta)
        # dA/dtheta * theta_dot @ v2 (only the v_B = R' v_W rows vary)
        dA = np.zeros((8, 4))
        dA[3, 0], dA[5, 0] = -s, c
        dA[3, 1], dA[5, 1] = -c, -s
        v2 = np.array([xd, zd, thd, ald])
        bias3_planar = A.T @ bias3 + A.T @ model.crba(q) @ (dA * thd) @ v2
        q2 = np.array([0.0, 0.0, theta, alpha])
        from fwmav_sim.dynamics.reduced_longitudinal import (
            ReducedModelParams,
            bias_vector,
            gravity_vector,
        )
        p2 = ReducedModelParams(m_t=_M_T, m_a=_M_A, I_t=_IT, I_a=_IA, d=_D, l_a=_L_A)
        bias2 = bias_vector(q2, v2, p2) + gravity_vector(q2, p2)
        np.testing.assert_allclose(bias3_planar, bias2, atol=1e-10)


def test_forward_dynamics_rnea_consistency():
    """M qdd = tau - bias identity: RNEA at the FD accelerations returns tau."""
    model = ArticulatedModel(_vehicle())
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = np.concatenate([
            rng.normal(size=3) * 0.05,
            quat_from_euler(*rng.uniform(-0.5, 0.5, 3)),
            rng.uniform(-0.4, 0.4, 2),
        ])
        u = rng.normal(size=8) * 0.3
        tau = rng.normal(size=8) * 1e-4
        qdd = model.forward_dynamics(q, u, tau)
        back = model.rnea(q, u, qdd)
        np.testing.assert_allclose(back, tau, rtol=1e-8, atol=1e-14)
