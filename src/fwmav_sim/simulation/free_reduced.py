"""Free reduced longitudinal model runner (M2 Task 8; also the foil plant).

Integrates the 8-state reduced model (dynamics/reduced_longitudinal.py) with
solve_ivp. Forces enter as body-frame point forces — from the cycle-averaged
injection (M1 record) or the foil's synthetic thrust law (labelled synthetic,
source: estimated; NOT physics).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
from fwmav_sim.core.timeseries import TimeSeries
from fwmav_sim.dynamics.reduced_longitudinal import (
    PointForce,
    ReducedAnalyticalModel,
    ReducedModelParams,
)
from fwmav_sim.vehicle.schema import SolverSpec


def synthetic_thrust_law_forces(k_thrust: float, base_f_hz: float,
                                freq_delta_hz: float, arm: float) -> list[PointForce]:
    """Foil force law (M2 §2.4a): T = k * f per thruster along -z_B at
    x = +/- arm. Frame mapping verified by the foil tests:
    T_front = k * f_L = k (base + delta), T_back = k * f_R = k (base - delta).
    Labelled synthetic — the sandbox's linear law is not physics."""
    f_l = max(0.0, base_f_hz + freq_delta_hz)
    f_r = max(0.0, base_f_hz - freq_delta_hz)
    t_front = k_thrust * f_l
    t_back = k_thrust * f_r
    return [((+arm, 0.0), (0.0, -t_front)), ((-arm, 0.0), (0.0, -t_back))]


def run_free_reduced(
    params: ReducedModelParams,
    solver: SolverSpec,
    initial_state: np.ndarray | None = None,
    forces: list[PointForce] | None = None,
    tau_act: float = 0.0,
    alpha_profile: AbdomenProfile | None = None,
    alpha_limits: tuple[float, float] | None = None,
    n_samples: int = 2001,
) -> TimeSeries:
    """Integrate the free reduced model; returns state/energy channels."""
    model = ReducedAnalyticalModel(params)
    forces = forces or []
    x0 = np.zeros(8) if initial_state is None else np.asarray(initial_state, float)

    def rhs(t, y):
        return model.rhs(y, t, forces=forces, tau_act=tau_act,
                         alpha_profile=alpha_profile, alpha_limits=alpha_limits)

    t_eval = np.linspace(0.0, solver.max_time, n_samples)
    sol = solve_ivp(rhs, (0.0, solver.max_time), x0,
                    method=solver.integrator, rtol=solver.rtol, atol=solver.atol,
                    t_eval=t_eval)
    if sol.status != 0:
        raise RuntimeError(f"free reduced solver failed: {sol.message}")

    energy = np.array([model.energy(y) for y in sol.y.T])
    channels = {
        "x": sol.y[0], "z": sol.y[1], "theta": sol.y[2], "alpha_abd": sol.y[3],
        "x_dot": sol.y[4], "z_dot": sol.y[5], "theta_dot": sol.y[6],
        "alpha_abd_dot": sol.y[7],
        "energy_total": energy,
    }
    meta = {
        "kind": "free_reduced",
        "solver": {"integrator": solver.integrator, "rtol": solver.rtol, "atol": solver.atol},
        "lock_abdomen": params.lock_abdomen,
        "n_point_forces": len(forces),
    }
    return TimeSeries(t=sol.t, channels=channels, meta=meta)
