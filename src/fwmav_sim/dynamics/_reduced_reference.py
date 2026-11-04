"""GENERATED FILE - do not edit by hand.

Regenerate with: python derivations/reduced_longitudinal.py
Source: docs/conventions/reduced_model_spec.md (Euler-Lagrange, SymPy).
"""

from __future__ import annotations

import math

import numpy as np


def free_M(theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a):
    """Mass matrix M(q) of the free reduced model; q = [x, z, theta, alpha_abd]."""
    return np.array([m_a + m_t, 0, m_a*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)), l_a*m_a*math.sin(alpha_abd + theta), 0, m_a + m_t, m_a*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)), l_a*m_a*math.cos(alpha_abd + theta), m_a*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)), m_a*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)), I_a + I_t + d**2*m_a + 2*d*l_a*m_a*math.cos(alpha_abd) + l_a**2*m_a, I_a + d*l_a*m_a*math.cos(alpha_abd) + l_a**2*m_a, l_a*m_a*math.sin(alpha_abd + theta), l_a*m_a*math.cos(alpha_abd + theta), I_a + d*l_a*m_a*math.cos(alpha_abd) + l_a**2*m_a, I_a + l_a**2*m_a], dtype=float).reshape(4, 4)

def free_bias(theta, alpha_abd, theta_d, alpha_abd_d, m_t, m_a, d, l_a):
    """Velocity-product (Coriolis/centrifugal) vector of the free model."""
    return np.array([m_a*(alpha_abd_d*l_a*(alpha_abd_d + theta_d)*math.cos(alpha_abd + theta) + theta_d*(alpha_abd_d*l_a*math.cos(alpha_abd + theta) + theta_d*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)))), -m_a*(alpha_abd_d*l_a*(alpha_abd_d + theta_d)*math.sin(alpha_abd + theta) + theta_d*(alpha_abd_d*l_a*math.sin(alpha_abd + theta) + theta_d*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)))), -alpha_abd_d*d*l_a*m_a*(alpha_abd_d + 2*theta_d)*math.sin(alpha_abd), d*l_a*m_a*theta_d**2*math.sin(alpha_abd)], dtype=float)

def free_G(theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a, g):
    """Gravity vector G(q) = dV/dq of the free model."""
    return np.array([0, -g*(m_a + m_t), -g*m_a*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)), -g*l_a*m_a*math.cos(alpha_abd + theta)], dtype=float)

def free_Q_point(theta, b_x, b_z, F_x_B, F_z_B):
    """Generalized force of a point force (F_x, F_z) in the BODY frame B applied at (b_x, b_z) in B."""
    return np.array([F_x_B*math.cos(theta) + F_z_B*math.sin(theta), -F_x_B*math.sin(theta) + F_z_B*math.cos(theta), F_x_B*b_z - F_z_B*b_x, 0], dtype=float)

def pend_M(phi_p, alpha_abd, m_t, m_a, I_t, I_a, u_tx, u_tz, j_x, j_z, l_a):
    """Mass matrix of the suspended-pendulum model; q_s = [phi_p, alpha_abd]."""
    return np.array([I_a + I_t + j_x**2*m_a - 2*j_x*l_a*m_a*math.cos(alpha_abd) + j_z**2*m_a + 2*j_z*l_a*m_a*math.sin(alpha_abd) + l_a**2*m_a + m_t*u_tx**2 + m_t*u_tz**2, I_a - j_x*l_a*m_a*math.cos(alpha_abd) + j_z*l_a*m_a*math.sin(alpha_abd) + l_a**2*m_a, I_a - j_x*l_a*m_a*math.cos(alpha_abd) + j_z*l_a*m_a*math.sin(alpha_abd) + l_a**2*m_a, I_a + l_a**2*m_a], dtype=float).reshape(2, 2)

def pend_bias(phi_p, alpha_abd, phi_p_d, alpha_abd_d, m_t, m_a, u_tx, u_tz, j_x, j_z, l_a):
    """Velocity-product vector of the pendulum model."""
    return np.array([alpha_abd_d*l_a*m_a*(alpha_abd_d*j_x*math.sin(alpha_abd) + alpha_abd_d*j_z*math.cos(alpha_abd) + 2*j_x*phi_p_d*math.sin(alpha_abd) + 2*j_z*phi_p_d*math.cos(alpha_abd)), -l_a*m_a*phi_p_d**2*(j_x*math.sin(alpha_abd) + j_z*math.cos(alpha_abd))], dtype=float)

def pend_G(phi_p, alpha_abd, m_t, m_a, I_t, I_a, u_tx, u_tz, j_x, j_z, l_a, g):
    """Gravity vector of the pendulum model."""
    return np.array([g*(m_a*(j_x*math.cos(phi_p) + j_z*math.sin(phi_p) - l_a*math.cos(alpha_abd + phi_p)) + m_t*(u_tx*math.cos(phi_p) + u_tz*math.sin(phi_p))), -g*l_a*m_a*math.cos(alpha_abd + phi_p)], dtype=float)

def free_A_lin_dG_dq(theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a, g):
    """dG/dq block for the free-model linearization."""
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g*m_a*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)), g*l_a*m_a*math.sin(alpha_abd + theta), 0, 0, g*l_a*m_a*math.sin(alpha_abd + theta), g*l_a*m_a*math.sin(alpha_abd + theta)], dtype=float).reshape(4, 4)

def free_A_lin_dQ_dq(theta, b_x, b_z, F_x_B, F_z_B):
    """dQ/dq block (body-frame point force) for the free-model linearization."""
    return np.array([0, 0, -F_x_B*math.sin(theta) + F_z_B*math.cos(theta), 0, 0, 0, -F_x_B*math.cos(theta) - F_z_B*math.sin(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float).reshape(4, 4)

def free_A_lin_dM_dtheta(theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a):
    """dM/dtheta block for the free-model linearization."""
    return np.array([0, 0, m_a*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)), l_a*m_a*math.cos(alpha_abd + theta), 0, 0, -m_a*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)), -l_a*m_a*math.sin(alpha_abd + theta), m_a*(d*math.cos(theta) + l_a*math.cos(alpha_abd + theta)), -m_a*(d*math.sin(theta) + l_a*math.sin(alpha_abd + theta)), 0, 0, l_a*m_a*math.cos(alpha_abd + theta), -l_a*m_a*math.sin(alpha_abd + theta), 0, 0], dtype=float).reshape(4, 4)

def free_A_lin_dM_dalpha(theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a):
    """dM/dalpha block for the free-model linearization."""
    return np.array([0, 0, l_a*m_a*math.cos(alpha_abd + theta), l_a*m_a*math.cos(alpha_abd + theta), 0, 0, -l_a*m_a*math.sin(alpha_abd + theta), -l_a*m_a*math.sin(alpha_abd + theta), l_a*m_a*math.cos(alpha_abd + theta), -l_a*m_a*math.sin(alpha_abd + theta), -2*d*l_a*m_a*math.sin(alpha_abd), -d*l_a*m_a*math.sin(alpha_abd), l_a*m_a*math.cos(alpha_abd + theta), -l_a*m_a*math.sin(alpha_abd + theta), -d*l_a*m_a*math.sin(alpha_abd), 0], dtype=float).reshape(4, 4)


def free_A_lin(theta_0, alpha_abd_0, b_x, b_z, F_x, F_z, tau_act,
               m_t, m_a, I_t, I_a, d, l_a, g):
    """Linearized state matrix A (8x8) of the free reduced model about
    (q0 = [x, z, theta_0, alpha_abd_0], qd = 0) with a point force (F_x, F_z)
    in the BODY frame B applied at (b_x, b_z) in B and actuator torque tau_act
    on alpha_abd.

    A = [[0, I], [M0^-1 (dQ/dq - dG/dq) - M0^-1 dM/dq_k M0^-1 h, 0]]
    (bias is quadratic in qd, so the velocity blocks vanish at qd = 0;
    damping, when configured, is added by the runtime model).
    """
    M0 = free_M(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a)
    M0inv = np.linalg.inv(M0)
    h = free_Q_point(theta_0, b_x, b_z, F_x, F_z) + np.array([0.0, 0.0, 0.0, tau_act]) \
        - free_G(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a, g)
    dG = free_A_lin_dG_dq(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a, g)
    dQ = free_A_lin_dQ_dq(theta_0, b_x, b_z, F_x, F_z)
    qdd0 = M0inv @ h
    dM = [free_A_lin_dM_dtheta(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a),
          free_A_lin_dM_dalpha(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a)]
    A = np.zeros((8, 8))
    A[:4, 4:] = np.eye(4)
    bottom = M0inv @ (dQ - dG)
    for k, dM in ((2, dM[0]), (3, dM[1])):
        bottom[:, k] += -M0inv @ dM @ qdd0
    A[4:, :4] = bottom
    return A

