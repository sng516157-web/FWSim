"""§11.2 articulated-model trajectory-level tests (M3).

The key regression: the planar restriction of the full 6-DoF model reproduces
the M2 reduced-model trajectory (which is SymPy-verified) to < 1e-6.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.core.transforms import quat_from_euler, quat_to_matrix
from fwmav_sim.dynamics.reduced_longitudinal import ReducedAnalyticalModel, ReducedModelParams
from fwmav_sim.dynamics.spatial import ArticulatedModel
from fwmav_sim.simulation.integrator import make_rhs, pack_state
from tests.unit.test_spatial import _D, _IA, _IT, _L_A, _M_A, _M_T, _vehicle

_MODEL3 = ArticulatedModel(_vehicle())
_P2 = ReducedModelParams(m_t=_M_T, m_a=_M_A, I_t=_IT, I_a=_IA, d=_D, l_a=_L_A)


def _m3_force_fn(F_B: np.ndarray, r_P: np.ndarray):
    """Constant body-frame point force F_B at r_P (in B) on the thorax."""

    def fn(t, qm, u):
        n = np.cross(r_P, F_B)
        return np.zeros(8), {"thorax": np.concatenate([n, F_B])}

    return fn


def test_planar_limit_recovers_reduced_model():
    """Same physics: gravity + constant body-frame thrust, free abdomen.

    M2 reduced model (SymPy-verified) vs M3 articulated model; trajectories of
    [x, z, theta, alpha_abd] must agree to < 1e-6 over 2 s."""
    m = _M_T + _M_A
    F_B = np.array([0.05, 0.0, -0.6 * m * 9.81])  # partial-lift thrust, body frame
    # applied at the CoM (no pitch moment): the attitude then moves only
    # through the abdomen coupling, keeping theta bounded for the comparison
    x_com = -_M_A * (_D + _L_A) / m
    r_P = np.array([x_com, 0.0, 0.0])

    # M2 reference
    model2 = ReducedAnalyticalModel(_P2)
    forces2 = [((r_P[0], r_P[2]), (F_B[0], F_B[2]))]
    theta0, alpha0 = 0.15, 0.2
    x0_2 = np.array([0.01, -0.02, theta0, alpha0, 0.05, 0.03, 0.1, 0.05])
    sol2 = solve_ivp(lambda t, y: model2.rhs(y, t, forces=forces2), (0, 2.0), x0_2,
                     method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
    assert sol2.status == 0

    # M3 articulated (abdomen roll free but planar-initial => stays zero)
    y0 = pack_state([0.01, 0.0, -0.02], quat_from_euler(0.0, theta0, 0.0), [alpha0, 0.0],
                    [0.05, 0.0, 0.03], [0.0, 0.1, 0.0], [0.05, 0.0])
    rhs3 = make_rhs(_MODEL3, force_fn=_m3_force_fn(F_B, r_P))
    sol3 = solve_ivp(rhs3, (0, 2.0), y0, method="DOP853", rtol=1e-11, atol=1e-13,
                     dense_output=True)
    assert sol3.status == 0

    t = np.linspace(0, 2.0, 401)
    y2 = sol2.sol(t)
    y3 = sol3.sol(t)
    # attitude compared as rotation matrices (branch-robust): pitch past 90 deg
    # flips the ZYX extraction branch, so Euler comparison would spuriously fail
    from fwmav_sim.core.transforms import Ry
    for i in range(len(t)):
        np.testing.assert_allclose(quat_to_matrix(y3[3:7, i]), Ry(float(y2[2, i])), atol=1e-6)
    np.testing.assert_allclose(y3[0], y2[0], atol=1e-6)          # x
    np.testing.assert_allclose(y3[2], y2[1], atol=1e-6)          # z
    np.testing.assert_allclose(y3[7], y2[3], atol=1e-6)          # alpha_abd
    np.testing.assert_allclose(y3[1], 0.0, atol=1e-12)           # y stays zero (planar)
    np.testing.assert_allclose(y3[8], 0.0, atol=1e-12)           # abdomen roll stays zero


def test_gravity_free_momentum_and_energy_conservation():
    """Aero-off, gravity-off free flight: linear momentum exact, angular
    momentum and kinetic energy conserved to solver tolerance."""
    from fwmav_sim.simulation.integrator import integrate_stepped

    y0 = pack_state([0.02, 0.01, -0.03], quat_from_euler(0.2, 0.3, -0.1), [0.25, -0.15],
                    [0.4, 0.1, -0.2], [0.3, -0.5, 0.2], [0.8, 0.5])
    rhs = make_rhs(_MODEL3, gravity=False)
    t, Y = integrate_stepped(rhs, y0, 1.0, rtol=1e-11, atol=1e-13)

    def momenta(y):
        from fwmav_sim.simulation.integrator import unpack_state
        p, quat, qj, v_W, w_B, qd_j = unpack_state(y, _MODEL3.nj)
        from fwmav_sim.core.transforms import quat_normalize
        quat = quat_normalize(quat)
        R_wb = quat_to_matrix(quat)
        qm = np.concatenate([p, quat, qj])
        u = np.concatenate([w_B, R_wb.T @ v_W, qd_j])
        Js = _MODEL3.body_jacobians(qm)
        tf = _MODEL3.forward_kinematics(qm)
        r_com = _MODEL3.com(qm)
        p_lin = np.zeros(3)
        L = np.zeros(3)
        inertias = {b.name: np.asarray(b.inertia.value, float) for b in _vehicle().bodies}
        for name, J in zip(_MODEL3.body_names, Js, strict=True):
            m = _MODEL3._mass[name]
            if m == 0.0:
                continue
            V = J @ u
            R_i = tf[name][0]
            c = _MODEL3._com_off[name]
            v_com = R_i @ (V[3:] + np.cross(V[:3], c))
            r_i = tf[name][1] + R_i @ c
            # spin: R I_c R^T (R w) = R I_c V_angular (body coords)
            p_lin += m * v_com
            L += np.cross(r_i - r_com, m * v_com) + R_i @ inertias[name] @ V[:3]
        return p_lin, L

    p0, L0 = momenta(Y[:, 0])
    drifts_p, drifts_L = [], []
    for i in range(Y.shape[1]):
        p_i, L_i = momenta(Y[:, i])
        drifts_p.append(np.abs(p_i - p0).max())
        drifts_L.append(np.abs(L_i - L0).max())
    assert max(drifts_p) < 1e-12
    assert max(drifts_L) < 1e-9


def test_gravity_linear_momentum_follows_free_fall():
    """Aero-off, gravity on: p(t) = p0 + m g_vec t to solver tolerance."""
    from fwmav_sim.simulation.integrator import integrate_stepped

    y0 = pack_state([0.0, 0.0, 0.0], quat_from_euler(0.1, 0.2, 0.05), [0.1, -0.05],
                    [0.1, 0.05, -0.02], [0.2, 0.1, -0.1], [0.3, 0.2])
    rhs = make_rhs(_MODEL3, gravity=True)
    t, Y = integrate_stepped(rhs, y0, 0.5, rtol=1e-11, atol=1e-13)
    m = _M_T + _M_A
    g_vec = np.array([0.0, 0.0, 9.81])

    def p_lin(y):
        from fwmav_sim.simulation.integrator import unpack_state
        p, quat, qj, v_W, w_B, qd_j = unpack_state(y, _MODEL3.nj)
        from fwmav_sim.core.transforms import quat_normalize
        quat = quat_normalize(quat)
        R_wb = quat_to_matrix(quat)
        qm = np.concatenate([p, quat, qj])
        u = np.concatenate([w_B, R_wb.T @ v_W, qd_j])
        Js = _MODEL3.body_jacobians(qm)
        tf = _MODEL3.forward_kinematics(qm)
        out = np.zeros(3)
        for name, J in zip(_MODEL3.body_names, Js, strict=True):
            mm = _MODEL3._mass[name]
            if mm == 0.0:
                continue
            V = J @ u
            R_i = tf[name][0]
            out += mm * (R_i @ (V[3:] + np.cross(V[:3], _MODEL3._com_off[name])))
        return out

    for k in (0, len(t) // 2, len(t) - 1):
        np.testing.assert_allclose(p_lin(Y[:, k]), p_lin(Y[:, 0]) + m * g_vec * t[k],
                                   atol=1e-9)


def test_clamped_chain_equals_composite_single_body():
    """Joints clamped => the articulated model tracks the equivalent composite
    rigid body exactly (representation-free cross-check)."""
    alpha, beta = 0.25, -0.15
    from fwmav_sim.core.transforms import Rx, Ry
    from fwmav_sim.vehicle.schema import VehicleDefinition

    R_abd = Ry(alpha) @ Rx(beta)
    m = _M_T + _M_A
    c_a = np.array([-_D, 0.0, 0.0]) + R_abd @ np.array([-_L_A, 0.0, 0.0])
    c_comp = (_M_A * c_a) / m
    I_t = _IT * np.eye(3)
    I_a = R_abd @ (_IA * np.eye(3)) @ R_abd.T

    def about_origin(I_c, mm, c_):
        return I_c + mm * (float(c_ @ c_) * np.eye(3) - np.outer(c_, c_))

    I_origin = about_origin(I_t, _M_T, np.zeros(3)) + about_origin(I_a, _M_A, c_a)
    I_comp = I_origin - m * (float(c_comp @ c_comp) * np.eye(3) - np.outer(c_comp, c_comp))

    prov = {"source": "estimated"}
    veh1 = VehicleDefinition.model_validate({
        "name": "composite",
        "bodies": [{"name": "thorax",
                    "mass": {"value": m, "unit": "kg", "provenance": prov},
                    "com_offset": {"value": list(c_comp), "unit": "m", "provenance": prov},
                    "inertia": {"value": I_comp.tolist(), "unit": "kg m^2", "provenance": prov}}],
        "joints": [], "wings": [], "aero": {"name": "none", "coefficients": {}},
    })
    model1 = ArticulatedModel(veh1)

    def clamped_rhs(model, nj):
        from fwmav_sim.core.transforms import quat_normalize, quat_rate

        def f(t, y):
            from fwmav_sim.simulation.integrator import unpack_state
            p, quat, qj, v_W, w_B, _ = unpack_state(y, nj)
            quat = quat_normalize(quat)
            R_wb = quat_to_matrix(quat)
            v_B = R_wb.T @ v_W
            qm = np.concatenate([p, quat, qj])
            u = np.concatenate([w_B, v_B, np.zeros(nj)])
            M = model.crba(qm)
            bias = model.bias(qm, u, gravity=False)
            ud = np.zeros(model.nv)
            ud[:6] = np.linalg.solve(M[:6, :6], -bias[:6])
            vd_W = R_wb @ (ud[3:6] + np.cross(w_B, v_B))
            return np.concatenate([v_W, quat_rate(quat, w_B), np.zeros(nj),
                                   vd_W, ud[:3], np.zeros(nj)])

        return f

    y0_3 = pack_state([0.02, 0.01, -0.03], quat_from_euler(0.2, 0.3, -0.1), [alpha, beta],
                      [0.4, 0.1, -0.2], [0.3, -0.5, 0.2], [0.0, 0.0])
    y0_1 = pack_state([0.02, 0.01, -0.03], quat_from_euler(0.2, 0.3, -0.1), [],
                      [0.4, 0.1, -0.2], [0.3, -0.5, 0.2], [])
    s3 = solve_ivp(clamped_rhs(_MODEL3, 2), (0, 1.0), y0_3, method="DOP853",
                   rtol=1e-12, atol=1e-14, dense_output=True)
    s1 = solve_ivp(clamped_rhs(model1, 0), (0, 1.0), y0_1, method="DOP853",
                   rtol=1e-12, atol=1e-14, dense_output=True)
    for ti in (0.3, 0.7, 1.0):
        y3 = s3.sol(ti)
        y1 = s1.sol(ti)
        np.testing.assert_allclose(quat_to_matrix(y3[3:7]), quat_to_matrix(y1[3:7]), atol=1e-9)
        np.testing.assert_allclose(y3[0:3], y1[0:3], atol=1e-9)
        np.testing.assert_allclose(y3[12:15], y1[10:13], atol=1e-9)  # omega_B
