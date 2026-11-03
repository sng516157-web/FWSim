"""§11.3 M2 numerical tests: pendulum energy drift and timestep refinement.

(Added in M2; docs/test_plan.md updated accordingly.)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
from fwmav_sim.constraints.suspended import SuspendedPendulum, SuspendedPendulumParams

_P = SuspendedPendulumParams(
    m_t=0.011, m_a=0.004, I_t=2e-6,
    u_tx=0.0, u_tz=0.05, j_x=-0.015, j_z=0.05, l_a=0.015,
)


def _equilibrium_shifted(p, lock=True):
    m = p.m_t + p.m_a
    x_com = (p.m_t * p.u_tx + p.m_a * (p.j_x - p.l_a)) / m
    return SuspendedPendulumParams(
        **{**p.__dict__, "u_tx": p.u_tx - x_com, "j_x": p.j_x - x_com,
           "lock_abdomen": lock}
    )


def test_frictionless_pendulum_energy_drift():
    """Energy drift < 1e-8 per period (DOP853, rtol 1e-9, atol 1e-12)."""
    p = _equilibrium_shifted(_P)
    pend = SuspendedPendulum(p)
    t_end = 10.0  # ~21 periods
    sol = solve_ivp(lambda t, y: pend.rhs(y, t), (0.0, t_end), [0.2, 0.0, 0.0, 0.0],
                    method="DOP853", rtol=1e-9, atol=1e-12, dense_output=True)
    t = np.linspace(0, t_end, 20001)
    E = np.array([pend.energy(y) for y in sol.sol(t).T])
    E0 = abs(E[0])
    n_periods = t_end / 0.4754  # measured period of this configuration
    drift_per_period = (E.max() - E.min()) / E0 / n_periods
    assert drift_per_period < 1e-8, f"drift per period: {drift_per_period:.3e}"


def test_actuated_pendulum_timestep_refinement():
    """max_step refinement on the actuated pendulum: the terminal state
    converges as max_step shrinks (DOP853 adaptive, rtol fixed)."""
    p = _equilibrium_shifted(_P, lock=False)
    pend = SuspendedPendulum(p)
    profile = AbdomenProfile(kind="sinusoid", amplitude=0.2, frequency=1.0)
    t_end = 4.0

    def final_state(max_step):
        sol = solve_ivp(lambda t, y: pend.rhs(y, t, alpha_profile=profile),
                        (0.0, t_end), np.zeros(4),
                        method="DOP853", rtol=1e-9, atol=1e-12, max_step=max_step)
        assert sol.status == 0
        return sol.y[:, -1]

    ref = final_state(1e-4)
    errs = []
    for ms in (0.1, 0.05, 0.01):
        errs.append(float(np.linalg.norm(final_state(ms) - ref) / np.linalg.norm(ref)))
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[2] < 1e-6
    print(f"\ntimestep refinement (max_step -> rel err vs 1e-4): "
          f"{dict(zip((0.1, 0.05, 0.01), errs, strict=True))}")
