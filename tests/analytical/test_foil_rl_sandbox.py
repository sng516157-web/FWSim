"""§11.2 foil tests — RL-sandbox dynamics-kernel cross-check (M2 §2.4a).

Cross-checks the reduced model in its single-rigid-body limit (abdomen
locked, m_a = 0) against Steven's 2D RL sandbox (docs/reference/rl_sandbox/reference_rl_flapping.py).
Expectations are computed from the sandbox's closed-form dynamics by hand,
never by running the sandbox code. The sandbox's T = k f thrust law is
synthetic — this validates the kernel (mass/inertia/force-law/lever-arm
assembly), nothing about aerodynamics.

Frame mapping (verified here, per the stage file's instruction not to assume):
    x_fs = x_sb,  z_fs = -y_sb,  theta_fs = -theta_sb
    T_front = k f_L = k (base_f + freq_delta)
    T_back  = k f_R = k (base_f - freq_delta)
The stage file's provisional "theta_fs == theta_sb, T_R <-> T_front" does NOT
hold: the sandbox's thrust direction (sin theta, cos theta) with y-up maps to
a sign-flipped pitch, and the torque sign follows. The tests below pin the
correct mapping; a sign error in any direction would fail them.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from fwmav_sim.dynamics.reduced_longitudinal import ReducedAnalyticalModel, ReducedModelParams
from fwmav_sim.simulation.free_reduced import synthetic_thrust_law_forces

# sandbox parameters (docs/reference/rl_sandbox/reference_rl_flapping.py)
M, G, I_PITCH, ARM, K = 0.05, 9.81, 2e-4, 0.1, 0.02
C_LIN, C_ANG = 0.05, 0.001

_PARAMS = ReducedModelParams(m_t=M, m_a=0.0, I_t=I_PITCH, d=0.0, l_a=0.0,
                             g=G, c_lin=C_LIN, c_ang=C_ANG, lock_abdomen=True)


def _sandbox_accel(x_sb, y_sb, th_sb, vx, vy, om, base_f, delta):
    """Hand-transcribed sandbox closed-form dynamics (unsteady terms off)."""
    f_l = max(0.0, base_f + delta)
    f_r = max(0.0, base_f - delta)
    T = K * (f_l + f_r)
    tau = ARM * K * (f_r - f_l)
    ax = (T * np.sin(th_sb) - C_LIN * vx) / M
    ay = (T * np.cos(th_sb) - M * G - C_LIN * vy) / M
    al = (tau - C_ANG * om) / I_PITCH
    return ax, ay, al


def _flapsim_accel(x_fs, z_fs, th_fs, vx, vz, om_fs, base_f, delta):
    model = ReducedAnalyticalModel(_PARAMS)
    forces = synthetic_thrust_law_forces(K, base_f, delta, ARM)
    state = np.array([x_fs, z_fs, th_fs, 0.0, vx, vz, om_fs, 0.0])
    xdot = model.rhs(state, 0.0, forces=forces)
    return xdot[4], xdot[5], xdot[6]


def test_foil_rhs_equivalence():
    """100 random states/inputs: FWSim accelerations == sandbox closed form
    under the verified mapping, to < 1e-12."""
    rng = np.random.default_rng(17)
    for _ in range(100):
        th_sb = rng.uniform(-0.5, 0.5)
        vx, vy = rng.uniform(-2, 2, 2)
        om_sb = rng.uniform(-5, 5)
        base_f = rng.uniform(0.0, 30.0)
        delta = rng.uniform(-5.0, 5.0)
        ax_sb, ay_sb, al_sb = _sandbox_accel(0.0, 0.0, th_sb, vx, vy, om_sb, base_f, delta)
        # mapping: theta_fs = -theta_sb, z = -y, omega_fs = -omega_sb
        ax_fs, az_fs, al_fs = _flapsim_accel(0.0, 0.0, -th_sb, vx, -vy, -om_sb, base_f, delta)
        assert abs(ax_fs - ax_sb) < 1e-12
        assert abs(az_fs + ay_sb) < 1e-12
        assert abs(al_fs + al_sb) < 1e-12


def test_foil_hover_equilibrium():
    """base_f = m g / (2 k) = 12.2625 Hz, delta = 0: equilibrium held over 10 s."""
    base_f = M * G / (2 * K)
    model = ReducedAnalyticalModel(_PARAMS)
    forces = synthetic_thrust_law_forces(K, base_f, 0.0, ARM)
    sol = solve_ivp(lambda t, y: model.rhs(y, t, forces=forces), (0.0, 10.0),
                    np.zeros(8), method="DOP853", rtol=1e-9, atol=1e-12)
    assert sol.status == 0
    assert np.abs(sol.y[0]).max() < 1e-9  # x
    assert np.abs(sol.y[1]).max() < 1e-9  # z
    assert np.abs(sol.y[2]).max() < 1e-12  # theta


def test_foil_pitch_response():
    """theta_0 = 0.05 rad perturbation, no control: compare against the
    closed-form solution of the sandbox equations (NOT the sandbox's Euler
    trajectory — dt = 0.02 Euler is itself an approximation).

    With delta = 0 the pitch torque is identically zero, so theta is constant
    and the translational ODEs are exactly linear with constant forcing —
    the closed form below is the exact solution, not just the linearization.
    """
    th_sb0 = 0.05
    base_f = M * G / (2 * K)
    T = M * G
    vx_star = T * np.sin(th_sb0) / C_LIN
    vy_star = (T * np.cos(th_sb0) - M * G) / C_LIN
    tau_x = M / C_LIN

    model = ReducedAnalyticalModel(_PARAMS)
    forces = synthetic_thrust_law_forces(K, base_f, 0.0, ARM)
    t_end = 2.0
    sol = solve_ivp(lambda t, y: model.rhs(y, t, forces=forces), (0.0, t_end),
                    np.array([0.0, 0.0, -th_sb0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    method="DOP853", rtol=1e-9, atol=1e-13, dense_output=True)
    t = np.linspace(0, t_end, 200)
    y = sol.sol(t)
    # theta constant at -theta_sb0 (mapping), omega zero
    np.testing.assert_allclose(y[2], -th_sb0, atol=1e-10)
    np.testing.assert_allclose(y[6], 0.0, atol=1e-12)
    # exact closed-form drift under the constant tilted thrust
    vx_exact = vx_star * (1 - np.exp(-t / tau_x))
    x_exact = vx_star * (t - tau_x * (1 - np.exp(-t / tau_x)))
    vz_exact = vy_star * (1 - np.exp(-t / tau_x))  # same tau: c_lin applies to both
    z_exact = -vy_star * (t - tau_x * (1 - np.exp(-t / tau_x)))  # z = -y
    np.testing.assert_allclose(y[4], vx_exact, rtol=1e-6)
    np.testing.assert_allclose(y[0], x_exact, rtol=1e-6)
    np.testing.assert_allclose(y[5], -vz_exact, rtol=1e-6)
    np.testing.assert_allclose(y[1], z_exact, rtol=1e-6)
