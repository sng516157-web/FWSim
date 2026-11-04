"""§11.2 concentrated-mass pendulum tests (M2 Task 5).

Passive frequency vs the analytical compound-pendulum reference, pendulum
NumPy model vs the SymPy reference, and the actuated-pendulum response.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
from fwmav_sim.constraints.suspended import (
    SuspendedPendulum,
    SuspendedPendulumParams,
    pend_bias,
    pend_gravity,
    pend_mass_matrix,
    pendulum_small_oscillation_frequency,
)
from fwmav_sim.dynamics import _reduced_reference as ref

# vehicle-like test parameters: pivot 50 mm above the thorax origin
_P = SuspendedPendulumParams(
    m_t=0.011, m_a=0.004, I_t=2e-6,
    u_tx=0.0, u_tz=0.05, j_x=-0.015, j_z=0.05, l_a=0.015,
)


def test_pendulum_matches_sympy_reference():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(100):
        phi, al = rng.uniform(-1.0, 1.0, 2)
        phid, ald = rng.uniform(-5.0, 5.0, 2)
        q_s = np.array([phi, al])
        qd_s = np.array([phid, ald])
        args = (_P.m_t, _P.m_a, _P.I_t, _P.I_a, _P.u_tx, _P.u_tz, _P.j_x, _P.j_z, _P.l_a)
        checks = (
            (pend_mass_matrix(q_s, _P), ref.pend_M(phi, al, *args)),
            (pend_bias(q_s, qd_s, _P), ref.pend_bias(phi, al, phid, ald,
                                                     _P.m_t, _P.m_a, _P.u_tx, _P.u_tz,
                                                     _P.j_x, _P.j_z, _P.l_a)),
            (pend_gravity(q_s, _P), ref.pend_G(phi, al, *args, _P.g)),
        )
        for got, want in checks:
            scale = max(float(np.abs(want).max()), 1e-12)
            worst = max(worst, float(np.abs(got - want).max()) / scale)
    assert worst < 1e-10


def _equilibrium_shifted(p: SuspendedPendulumParams) -> SuspendedPendulumParams:
    """Shift the pivot horizontally so the CoM hangs directly below it
    (alpha = 0 level equilibrium)."""
    m = p.m_t + p.m_a
    x_com = (p.m_t * p.u_tx + p.m_a * (p.j_x - p.l_a)) / m
    return SuspendedPendulumParams(
        **{**p.__dict__, "u_tx": p.u_tx - x_com, "j_x": p.j_x - x_com,
           "lock_abdomen": True}
    )


def test_passive_pendulum_frequency():
    """Small-oscillation frequency vs omega = sqrt(m g d_cm / I_pivot),
    relative error < 1e-6.

    Measurement: solver event detection on downward zero crossings (accurate
    to solver tolerance); release angle 1e-3 rad keeps the finite-amplitude
    period correction (phi0^2/16 = 6.25e-8) below the tolerance.
    """
    p = _equilibrium_shifted(_P)
    pend = SuspendedPendulum(p)
    omega_expected = pendulum_small_oscillation_frequency(p)

    def zero_crossing(t, y):
        return y[0]

    zero_crossing.direction = -1  # same-direction crossings: full period
    sol = solve_ivp(
        lambda t, y: pend.rhs(y, t), (0.0, 10.0), [0.001, 0.0, 0.0, 0.0],
        method="DOP853", rtol=1e-9, atol=1e-12, dense_output=True,
        events=zero_crossing,
    )
    periods = np.diff(sol.t_events[0])
    omega_measured = 2 * np.pi / periods.mean()
    assert abs(omega_measured - omega_expected) / omega_expected < 1e-6


def test_actuated_pendulum_response():
    """Prescribed sinusoidal abdomen motion under suspension.

    Checks: (1) actual alpha tracks the command exactly (prescribed mode);
    (2) the swing responds (phi departs from equilibrium); (3) the recorded
    required actuator torque is finite and consistent with the free-mode
    residual M qdd + bias + G - Q at the computed accelerations.
    """
    p = SuspendedPendulumParams(**{**_P.__dict__})
    pend = SuspendedPendulum(p)
    profile = AbdomenProfile(kind="sinusoid", amplitude=0.1, frequency=0.5)
    states: list = []

    def rhs(t, y):
        xdot = pend.rhs(y, t, alpha_profile=profile)
        states.append((t, pend.last_tau_act_required, y[1]))
        return xdot

    sol = solve_ivp(rhs, (0.0, 4.0), [0.0, 0.0, 0.0, 0.0],
                    method="DOP853", rtol=1e-9, atol=1e-12, dense_output=True)
    assert sol.status == 0
    t = np.linspace(0, 4.0, 4001)
    y = sol.sol(t)
    np.testing.assert_allclose(y[1], 0.1 * np.sin(np.pi * t), atol=1e-12)
    assert np.abs(y[0]).max() > 1e-4  # swing responds to the abdomen motion

    # torque consistency: residual of the alpha equation with recorded tau
    for (ti, tau_req, _) in states[:: max(1, len(states) // 50)]:
        yd = pend.rhs(sol.sol(ti), ti, alpha_profile=profile)
        q_s = np.array([sol.sol(ti)[0], 0.1 * np.sin(np.pi * ti)])
        qd_s = np.array([sol.sol(ti)[2], 0.1 * np.pi * np.cos(np.pi * ti)])
        qdd_s = yd[2:]
        M = pend_mass_matrix(q_s, pend.p)
        resid = (M @ qdd_s + pend_bias(q_s, qd_s, pend.p) + pend_gravity(q_s, pend.p))[1]
        assert abs(resid - tau_req) < 1e-9
