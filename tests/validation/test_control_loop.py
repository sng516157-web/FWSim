"""Control design-loop tests (M6 Task 8): trim, linearize, LQR, closed loop."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.control.closed_loop import run_closed_loop
from fwmav_sim.control.linearize import linearize_trim
from fwmav_sim.control.lqr import lqr_gain
from fwmav_sim.control.trim import solve_trim
from fwmav_sim.dynamics.reduced_longitudinal import ReducedModelParams

_P = ReducedModelParams(m_t=0.011, m_a=0.004, I_t=2e-6, I_a=5e-7, d=0.015, l_a=0.015)
_MG = (_P.m_t + _P.m_a) * _P.g
_R_T = (0.0, 0.0)


def test_trim_balances_weight_and_moment():
    # cycle-averaged force too weak: trim must scale it up to lift = weight
    F_avg = (0.001, -0.8 * _MG)
    trim = solve_trim(F_avg, _R_T, _P)
    assert trim.thrust_scale == pytest.approx(1.25, rel=1e-3)
    assert abs(trim.residual[0]) < 1e-9  # lift balance
    assert abs(trim.residual[1]) < 1e-9  # moment balance
    assert trim.force_trim[1] == pytest.approx(-_MG, rel=1e-9)


def test_lqr_stabilizes_linearization():
    trim = solve_trim((0.001, -0.8 * _MG), _R_T, _P)
    A, B = linearize_trim(float(trim.force_trim[0]), float(trim.force_trim[1]),
                          trim.r_T_trim, trim.alpha_abd_trim, trim.tau_act_trim, _P)
    K = lqr_gain(A, B, np.diag([1, 10, 50, 5, 1, 1, 5, 1]), np.eye(2))
    eigs = np.linalg.eigvals(A - B @ K)
    assert np.all(np.real(eigs) < 0)


def test_closed_loop_converges_from_perturbation():
    trim = solve_trim((0.001, -0.8 * _MG), _R_T, _P)
    x0 = np.array([0.0, -0.005, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0])
    res = run_closed_loop(_P, trim, trim.r_T_trim,
                          Q=np.diag([1, 10, 50, 5, 1, 1, 5, 1]), R=np.eye(2),
                          x0=x0, t_end=3.0, n_samples=601)
    ts = res.ts
    x_trim_alpha = trim.alpha_abd_trim
    # perturbations decay: theta and z settle to the trim neighbourhood
    assert abs(ts["theta"][-1]) < 0.02
    assert abs(ts["z"][-1] - ts["z"][0]) < 0.005
    assert abs(ts["alpha_abd"][-1] - x_trim_alpha) < 0.05
    # control signals finite and bounded (no bang-bang blowup)
    assert np.abs(ts["du_thrust"]).max() < 5.0
    assert np.abs(ts["du_tau"]).max() < 1.0
