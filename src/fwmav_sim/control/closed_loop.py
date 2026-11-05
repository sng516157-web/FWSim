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
                    rtol: float = 1e-9, atol: float = 1e-12) -> ClosedLoopResult:
    """Simulate the LQR-regulated reduced model from x0 back to trim."""
    p = params
    F_x, F_z = float(trim.force_trim[0]), float(trim.force_trim[1])
    A, B = linearize_trim(F_x, F_z, r_T, trim.alpha_abd_trim, trim.tau_act_trim, p)
    K = lqr_gain(A, B, Q, R)
    model = ReducedAnalyticalModel(p)
    x_trim = np.array([0.0, 0.0, 0.0, trim.alpha_abd_trim, 0.0, 0.0, 0.0, 0.0])

    def rhs(t, x):
        du = -K @ (x - x_trim)
        tau = trim.tau_act_trim + du[1]
        forces = [(r_T, ((1.0 + du[0]) * F_x, (1.0 + du[0]) * F_z))]
        return model.rhs(x, t, forces=forces, tau_act=tau)

    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(x0, float), method="DOP853",
                    rtol=rtol, atol=atol, t_eval=np.linspace(0.0, t_end, n_samples))
    if sol.status != 0:
        raise RuntimeError(f"closed-loop solver failed: {sol.message}")
    channels = {
        "x": sol.y[0], "z": sol.y[1], "theta": sol.y[2], "alpha_abd": sol.y[3],
        "x_dot": sol.y[4], "z_dot": sol.y[5], "theta_dot": sol.y[6],
        "alpha_abd_dot": sol.y[7],
        "du_thrust": np.array([-(K @ (y - x_trim))[0] for y in sol.y.T]),
        "du_tau": np.array([-(K @ (y - x_trim))[1] for y in sol.y.T]),
    }
    meta = {"kind": "closed_loop_reduced", "controller": "lqr",
            "note": "design-loop demo on the Level-1 model; NOT flight-ready (A-007)"}
    return ClosedLoopResult(ts=TimeSeries(t=sol.t, channels=channels, meta=meta),
                            K=K, trim=trim)
