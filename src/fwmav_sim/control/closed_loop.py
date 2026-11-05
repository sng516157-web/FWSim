"""Closed-loop hover demo on the reduced model (M6 Task 8).

Plant: the nonlinear 8-state reduced model (dynamics/reduced_longitudinal).
Controller: LQR over the trim linearization, regulating the state back to
the trim point from a perturbed initial condition. Control channels:
[delta_thrust_scale, delta_tau_act].

This is a design-loop demo (trim -> linearize -> design -> simulate), NOT a
flight-ready autopilot: the plant is Level 1 (cycle-averaged forces, A-007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.control.linearize import linearize_trim
from fwmav_sim.control.lqr import lqr_gain
from fwmav_sim.control.trim import TrimResult
from fwmav_sim.core.timeseries import TimeSeries
from fwmav_sim.dynamics.reduced_longitudinal import (
    ReducedAnalyticalModel,
    ReducedModelParams,
)


@dataclass(frozen=True)
class ClosedLoopResult:
    ts: TimeSeries
    K: np.ndarray  # LQR gain (2x8)
    trim: TrimResult


def run_closed_loop(params: ReducedModelParams, trim: TrimResult,
                    r_T: tuple[float, float],
                    Q: np.ndarray, R: np.ndarray,
                    x0: np.ndarray, t_end: float, n_samples: int = 2001,
                    rtol: float = 1e-9, atol: float = 1e-12,
                    control_scale: tuple[float, float] = (1.0, 1e-3),
                    damping_D: np.ndarray | None = None,
                    du_limits: tuple[float, float] | None = None) -> ClosedLoopResult:
    """Simulate the LQR-regulated reduced model from x0 back to trim.

    control_scale normalizes the control channels before the LQR solve
    (du = S du_hat): channel 0 is dimensionless (thrust scale ~1) while
    channel 1 is a torque (~1e-4 N m physical). Without scaling, the LQR
    treats 1 N m as one unit of effort and commands torques ~10^6 times the
    trim torque, which detonates the tiny abdomen inertia (M6 bug 3).

    damping_D: cycle-averaged aerodynamic body damping (dynamics/aero_damping
    — flapping counter-torque + heave damping; the physically dominant hover
    stabilization, absent from a constant-force model).
    du_limits: optional actuator saturation (|du| per channel).
    """
    p = params
    F_x, F_z = float(trim.force_trim[0]), float(trim.force_trim[1])
    A, B = linearize_trim(F_x, F_z, r_T, trim.theta_trim, trim.alpha_abd_trim,
                          trim.tau_act_trim, p, damping_D=damping_D)
    S = np.diag(control_scale)
    K_hat = lqr_gain(A, B @ S, Q, R)
    K = S @ K_hat  # physical-units gain: du = -K (x - x_trim)
    model = ReducedAnalyticalModel(p)
    x_trim = np.array([0.0, 0.0, trim.theta_trim, trim.alpha_abd_trim,
                       0.0, 0.0, 0.0, 0.0])

    def du_applied(x):
        du = -K @ (x - x_trim)
        if du_limits is not None:
            du = np.clip(du, [-du_limits[0], -du_limits[1]], du_limits)
        return du

    def rhs(t, x):
        du = du_applied(x)
        tau = trim.tau_act_trim + du[1]
        forces = [(r_T, ((1.0 + du[0]) * F_x, (1.0 + du[0]) * F_z))]
        q_extra = damping_D @ x[4:] if damping_D is not None else None
        return model.rhs(x, t, forces=forces, tau_act=tau, q_extra=q_extra)

    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(x0, float), method="DOP853",
                    rtol=rtol, atol=atol, t_eval=np.linspace(0.0, t_end, n_samples))
    if sol.status != 0:
        raise RuntimeError(f"closed-loop solver failed: {sol.message}")
    channels = {
        "x": sol.y[0], "z": sol.y[1], "theta": sol.y[2], "alpha_abd": sol.y[3],
        "x_dot": sol.y[4], "z_dot": sol.y[5], "theta_dot": sol.y[6],
        "alpha_abd_dot": sol.y[7],
        "du_thrust": np.array([du_applied(y)[0] for y in sol.y.T]),
        "du_tau": np.array([du_applied(y)[1] for y in sol.y.T]),
    }
    meta = {"kind": "closed_loop_reduced", "controller": "lqr",
            "note": "design-loop demo on the Level-1 model; NOT flight-ready (A-007)"}
    return ClosedLoopResult(ts=TimeSeries(t=sol.t, channels=channels, meta=meta),
                            K=K, trim=trim)
