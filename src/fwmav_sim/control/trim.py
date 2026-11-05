"""Hover trim for the reduced longitudinal model (M6 Task 8).

Trim problem (reduced_model_spec.md §6): find (thrust_scale, alpha_abd_trim,
x_T_shift, theta_trim) such that the cycle-averaged wing force, scaled by
thrust_scale, rotated by theta_trim, and applied at r_T shifted aft by
x_T_shift, balances weight, and the pitch moment about the instantaneous CoM
vanishes:

    [Ry(theta) (k F_avg)]_x                       = 0   (no horizontal force)
    [Ry(theta) (k F_avg)]_z + (m_t + m_a) g       = 0   (lift = weight)
    moment about CoM(k F_avg at r_T + dx, alpha)  = 0

alpha_abd is BOUNDED (default: the abdomen joint limits of the reference
vehicle, ±45 deg) — an unbounded solve folds the abdomen fully forward
(alpha ~ pi) to drag the CoM under the wing resultant, which is a
mathematically valid but physically absurd, dynamically unstable "trim".
x_T_shift is the design variable that takes over instead: it is the wing
placement (centre-of-pressure to CoM alignment) a designer actually tunes,
and the closed-loop plant applies the force at the shifted point.

NOT a free-body trim (Level 2): the reduced model's trim is the Level-1
entry point for the control loop; the inputs are the cycle-averaged forces
from a time-resolved M1 run (A-007 — same honesty chain as M2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from fwmav_sim.dynamics.reduced_longitudinal import ReducedModelParams


@dataclass(frozen=True)
class TrimResult:
    """Solved hover trim for the reduced model."""

    thrust_scale: float  # k applied to the cycle-averaged force vector
    alpha_abd_trim: float  # rad, abdomen angle at trim (bounded)
    x_T_shift: float  # m, aft shift of the force application point (design trim)
    theta_trim: float  # rad, body pitch at trim (balances the x force)
    tau_act_trim: float  # actuator torque needed to hold alpha_abd_trim
    force_trim: np.ndarray  # (2,) scaled (F_x, F_z) in B at trim
    r_T_trim: tuple[float, float]  # application point actually used by the plant
    residual: np.ndarray  # (3,) [Fx balance, Fz balance, moment] at the solution


def _moment_about_com(alpha: float, F_x: float, F_z: float, r_T: tuple[float, float],
                      p: ReducedModelParams) -> float:
    """Pitch moment of the wing resultant about the instantaneous CoM (z down,
    y positive; spec §4 kinematics at theta = 0)."""
    x_T, z_T = r_T
    # CoM in B at alpha: thorax at the origin; abdomen CoM at
    # (-d - l_a cos alpha, +l_a sin alpha) (spec §4 with theta = 0)
    x_com = -p.m_a * (p.d + p.l_a * np.cos(alpha)) / (p.m_t + p.m_a)
    z_com = p.m_a * p.l_a * np.sin(alpha) / (p.m_t + p.m_a)
    # moment of F at r_T about CoM (2D cross product, y-component)
    return (x_T - x_com) * F_z - (z_T - z_com) * F_x


def solve_trim(F_avg: tuple[float, float], r_T: tuple[float, float],
               params: ReducedModelParams,
               alpha_bounds: tuple[float, float] = (-0.785398, 0.785398),
               x0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> TrimResult:
    """Solve the hover trim. F_avg: cycle-averaged (F_x, F_z) in B at the
    thrust application point r_T (from an M1 record, A-007)."""
    m_g = (params.m_t + params.m_a) * params.g

    def residual(x):
        k, alpha, dx, theta = x
        F_x, F_z = k * F_avg[0], k * F_avg[1]
        # world-frame force balance (body force rotated by theta; z down)
        fw_x = np.cos(theta) * F_x + np.sin(theta) * F_z
        fw_z = -np.sin(theta) * F_x + np.cos(theta) * F_z
        return [fw_x, fw_z + m_g,
                _moment_about_com(alpha, F_x, F_z, (r_T[0] + dx, r_T[1]), params)]

    sol = least_squares(residual, x0,
                        bounds=([-np.inf, alpha_bounds[0], -np.inf, -np.inf],
                                [np.inf, alpha_bounds[1], np.inf, np.inf]))
    k, alpha, dx, theta = (float(v) for v in sol.x)
    F_x, F_z = k * F_avg[0], k * F_avg[1]
    # actuator torque holding alpha against the abdomen's gravity moment:
    # equilibrium of the reduced model requires Q[3] = G[3] with
    # G[3] = -m_a g l_a cos(theta + alpha) (spec §5: gravity drives alpha
    # positive, tip downward; the actuator opposes it). Getting this sign
    # wrong makes the "trim" a non-equilibrium and the closed loop diverges
    # (M6 bug).
    tau = -params.m_a * params.g * params.l_a * np.cos(theta + alpha)
    return TrimResult(thrust_scale=k, alpha_abd_trim=alpha, x_T_shift=dx,
                      theta_trim=theta, tau_act_trim=tau,
                      force_trim=np.array([F_x, F_z]),
                      r_T_trim=(r_T[0] + dx, r_T[1]),
                      residual=np.asarray(sol.fun))
