"""Linearization of the reduced model at trim (M6 Task 8).

Uses the SymPy-derived analytic path (_reduced_reference.free_A_lin) for the
state matrix — the route M2 already verified against numeric Jacobians — and
builds the input matrix for the control channels

    u = [delta_thrust_scale, delta_tau_act]

i.e. perturbations of the wing-force scale and the abdomen actuator torque
about their trim values.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.dynamics import _reduced_reference as ref
from fwmav_sim.dynamics.reduced_longitudinal import ReducedModelParams


def linearize_trim(F_x: float, F_z: float, r_T: tuple[float, float],
                   theta_trim: float, alpha_trim: float, tau_trim: float,
                   params: ReducedModelParams,
                   damping_D: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of the 8-state reduced model at a hover trim point.

    damping_D: optional (4,4) generalized damping (dynamics/aero_damping.py):
    the cycle-averaged aerodynamic body damping — flapping counter-torque and
    heave damping. WITHOUT it the hover trim is open-loop unstable in the
    nonlinear plant and no LQR demo converges (M6 report §3).
    """
    p = params
    A = ref.free_A_lin(theta_trim, alpha_trim, r_T[0], r_T[1], F_x, F_z, tau_trim,
                       p.m_t, p.m_a, p.I_t, p.I_a, p.d, p.l_a, p.g)
    M0 = ref.free_M(theta_trim, alpha_trim, p.m_t, p.m_a, p.I_t, p.I_a, p.d, p.l_a)
    M0inv = np.linalg.inv(M0)
    if damping_D is not None:
        A[4:, 4:] += M0inv @ np.asarray(damping_D, dtype=float)
    # B: du1 scales the point force; du2 adds actuator torque on channel 3
    Q_unit = ref.free_Q_point(theta_trim, r_T[0], r_T[1], F_x, F_z)
    B = np.zeros((8, 2))
    B[4:, 0] = M0inv @ Q_unit
    B[4:, 1] = M0inv @ np.array([0.0, 0.0, 0.0, 1.0])
    return A, B
