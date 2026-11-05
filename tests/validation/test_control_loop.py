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
    assert trim.thrust_scale == pytest.approx(1.25, rel=1e-2)
    assert np.abs(trim.residual).max() < 1e-8  # Fx, Fz and moment balances
    assert abs(trim.alpha_abd_trim) <= 0.785398 + 1e-9  # bounded (M6 bug fix)
    # the trim is an equilibrium of the NONLINEAR plant (Q == G)
    from fwmav_sim.dynamics.reduced_longitudinal import (
        generalized_force,
        gravity_vector,
    )

    q = np.array([0.0, 0.0, trim.theta_trim, trim.alpha_abd_trim])
    Q = generalized_force(trim.theta_trim,
                          [(trim.r_T_trim, (trim.force_trim[0], trim.force_trim[1]))])
    Q[3] += trim.tau_act_trim
    np.testing.assert_allclose(Q, gravity_vector(q, _P), atol=1e-7)


def test_lqr_stabilizes_linearization():
    trim = solve_trim((0.001, -0.8 * _MG), _R_T, _P)
    A, B = linearize_trim(float(trim.force_trim[0]), float(trim.force_trim[1]),
                          trim.r_T_trim, trim.theta_trim, trim.alpha_abd_trim,
                          trim.tau_act_trim, _P)
    K = lqr_gain(A, B, np.diag([1, 10, 50, 5, 1, 1, 5, 1]), np.eye(2))
    eigs = np.linalg.eigvals(A - B @ K)
    assert np.all(np.real(eigs) < 0)


def test_closed_loop_converges_from_perturbation():
    """With the cycle-averaged aerodynamic body damping (flapping
    counter-torque), the LQR holds a 0.2 rad perturbation (M6 §3: without
    damping the trim is open-loop unstable and every variant diverges)."""
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.dynamics.aero_damping import body_motion_damping, damping_matrix
    from fwmav_sim.experiment.config import load_experiment, load_vehicle

    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    kin_map = {w.name: exp.kinematics for w in vehicle.wings}
    d = body_motion_damping(vehicle, kin_map, env)
    # physically dominant damping, correct signs
    assert d["dFz_dvz"] < 0.0 and d["dMy_dwy"] < 0.0
    D = damping_matrix(d)

    trim = solve_trim((0.001, -0.8 * _MG), _R_T, _P)
    x0 = np.array([0.0, -0.005, trim.theta_trim + 0.2, 0.05, 0.0, 0.0, 0.0, 0.0])
    res = run_closed_loop(_P, trim, trim.r_T_trim,
                          Q=np.diag([1, 10, 50, 5, 1, 1, 5, 1]), R=np.eye(2),
                          x0=x0, t_end=3.0, n_samples=601,
                          damping_D=D, du_limits=(0.5, 2e-3))
    ts = res.ts
    # perturbations decay: theta and z settle to the trim neighbourhood
    assert abs(ts["theta"][-1] - trim.theta_trim) < 0.02
    assert abs(ts["z"][-1] - ts["z"][0]) < 0.05
    assert abs(ts["alpha_abd"][-1] - trim.alpha_abd_trim) < 0.1
    # control signals finite and bounded (no bang-bang blowup)
    assert np.abs(ts["du_thrust"]).max() <= 0.5 + 1e-12
    assert np.abs(ts["du_tau"]).max() <= 2e-3 + 1e-12
