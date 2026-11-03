"""§11.2 reduced thorax-abdomen model tests (M2 Tasks 4).

- NumPy runtime vs SymPy-derived reference at 100 random states (< 1e-10)
- zero-abdomen-motion baseline (rigid limit)
- linearized state matrix vs numerical Jacobian of the nonlinear rhs
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.dynamics import _reduced_reference as ref
from fwmav_sim.dynamics.reduced_longitudinal import (
    ReducedAnalyticalModel,
    ReducedModelParams,
    bias_vector,
    gravity_vector,
    mass_matrix,
)

_P = ReducedModelParams(m_t=0.011, m_a=0.004, I_t=2e-6, d=0.015, l_a=0.015)


def test_reduced_model_matches_sympy_reference():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        th, al = rng.uniform(-1.0, 1.0, 2)
        thd, ald = rng.uniform(-5.0, 5.0, 2)
        q = np.array([0.0, 0.0, th, al])
        qd = np.array([0.0, 0.0, thd, ald])
        checks = (
            (mass_matrix(q, _P), ref.free_M(th, al, _P.m_t, _P.m_a, _P.I_t, _P.d, _P.l_a)),
            (bias_vector(q, qd, _P), ref.free_bias(th, al, thd, ald, _P.m_t, _P.m_a, _P.d, _P.l_a)),
            (gravity_vector(q, _P),
             ref.free_G(th, al, _P.m_t, _P.m_a, _P.I_t, _P.d, _P.l_a, _P.g)),
        )
        for got, want in checks:
            scale = max(float(np.abs(want).max()), 1e-12)
            worst = max(worst, float(np.abs(got - want).max()) / scale)
    assert worst < 1e-10


def test_zero_abdomen_motion_baseline():
    """Rigid limit: abdomen locked at 0 => single-body planar dynamics.

    With a point force through the CoM and no damping, the centre of mass must
    accelerate exactly along the applied force and pitch/abdomen accelerations
    vanish."""
    p = ReducedModelParams(m_t=0.011, m_a=0.004, I_t=2e-6, d=0.015, l_a=0.015,
                           lock_abdomen=True)
    model = ReducedAnalyticalModel(p)
    # CoM x-offset at alpha = 0 (thorax at origin)
    x_com = -p.m_a * (p.d + p.l_a) / (p.m_t + p.m_a)
    state = np.zeros(8)
    F = ((x_com, 0.0), (0.3, -0.2))  # force through the CoM, body frame
    xdot = model.rhs(state, 0.0, forces=[F])
    m = p.m_t + p.m_a
    np.testing.assert_allclose(xdot[4], 0.3 / m, rtol=1e-12)
    np.testing.assert_allclose(xdot[5], -0.2 / m + p.g, rtol=1e-12)
    assert abs(xdot[6]) < 1e-15  # no pitch acceleration
    assert xdot[7] == 0.0  # locked abdomen


def test_linearized_vs_numerical_jacobian():
    """SymPy-derived A_lin vs central FD Jacobian of the NumPy rhs.

    Hover trim: theta_0 = 0, alpha = 0, thrust force balancing weight applied
    at the CoM (no pitch moment), zero damping (A_lin carries no damping).
    """
    m = _P.m_t + _P.m_a
    x_com = -_P.m_a * (_P.d + _P.l_a) / m
    F0 = (0.0, m * _P.g)  # (F_x, F_z) in W = lift (z down); body level
    # generalized_force takes body-frame forces; at theta=0 body == world
    forces = [((x_com, 0.0), F0)]
    A_ref = ref.free_A_lin(0.0, 0.0, x_com, 0.0, *F0, 0.0,
                           _P.m_t, _P.m_a, _P.I_t, _P.d, _P.l_a, _P.g)

    model = ReducedAnalyticalModel(_P)
    x0 = np.zeros(8)
    eps = 1e-7
    A_fd = np.zeros((8, 8))
    for j in range(8):
        dx = np.zeros(8)
        dx[j] = eps
        fp = model.rhs(x0 + dx, 0.0, forces=forces)
        fm = model.rhs(x0 - dx, 0.0, forces=forces)
        A_fd[:, j] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(A_fd, A_ref, rtol=1e-5, atol=1e-8)
