"""Suspended-pendulum runner (M2 Task 8).

Solves the rigid-pivot constraint model (constraints/suspended.py) with
solve_ivp. Records commanded vs actual abdomen angle (identical in prescribed
mode by construction), the required actuator torque, and the total mechanical
energy per sample.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha
from fwmav_sim.constraints.suspended import SuspendedPendulum, SuspendedPendulumParams
from fwmav_sim.core.timeseries import TimeSeries
from fwmav_sim.vehicle.schema import SolverSpec


def run_suspended_pendulum(
    params: SuspendedPendulumParams,
    solver: SolverSpec,
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    alpha_profile: AbdomenProfile | None = None,
    alpha_limits: tuple[float, float] | None = None,
    n_samples: int = 2001,
) -> TimeSeries:
    """Integrate the suspended pendulum; returns state/energy channels."""
    pend = SuspendedPendulum(params)

    def rhs(t, y):
        return pend.rhs(y, t, alpha_profile=alpha_profile, alpha_limits=alpha_limits)

    t_eval = np.linspace(0.0, solver.max_time, n_samples)
    sol = solve_ivp(rhs, (0.0, solver.max_time), np.asarray(initial_state, float),
                    method=solver.integrator, rtol=solver.rtol, atol=solver.atol,
                    t_eval=t_eval)
    if sol.status != 0:
        raise RuntimeError(f"pendulum solver failed: {sol.message}")

    # per-sample channels, recomputed on the output grid
    alpha_cmd = np.zeros(len(sol.t))
    tau_req = np.zeros(len(sol.t))
    energy = np.zeros(len(sol.t))
    for i, (ti, y) in enumerate(zip(sol.t, sol.y.T, strict=True)):
        if alpha_profile is not None:
            alpha_cmd[i] = prescribed_alpha(alpha_profile, ti, alpha_limits)[0]
        else:
            alpha_cmd[i] = y[1]
        pend.rhs(y, ti, alpha_profile=alpha_profile, alpha_limits=alpha_limits)
        tau_req[i] = pend.last_tau_act_required
        energy[i] = pend.energy(y)

    channels = {
        "phi_p": sol.y[0],
        "alpha_abd": sol.y[1],
        "phi_p_dot": sol.y[2],
        "alpha_abd_dot": sol.y[3],
        "alpha_commanded": alpha_cmd,
        "tau_act_required": tau_req,
        "energy_total": energy,
    }
    meta = {
        "kind": "suspended_pendulum",
        "solver": {"integrator": solver.integrator, "rtol": solver.rtol, "atol": solver.atol},
        "abdomen_mode": ("prescribed" if alpha_profile
                         else "locked" if params.lock_abdomen else "free"),
    }
    return TimeSeries(t=sol.t, channels=channels, meta=meta)
