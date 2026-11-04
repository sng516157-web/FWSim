"""§11.1 actuator tests.

Real (M2): prescribed abdomen profile derivatives. Stubs (M3): profile rate
limiting and torque saturation recording (actuator stages 2-3).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha


def test_profile_derivatives_finite_difference():
    profiles = [
        AbdomenProfile(kind="sinusoid", amplitude=0.3, frequency=3.0, phase=0.4, offset=0.05),
        AbdomenProfile(kind="ramp", amplitude=0.2, t_start=0.1, t_end=0.6, offset=0.02),
        AbdomenProfile(kind="step", amplitude=0.1, t_start=0.2, offset=0.01),
    ]
    for p in profiles:
        for t in np.linspace(0.02, 0.95, 25):
            if p.kind == "step" and abs(t - p.t_start) < 1e-3:
                continue  # discontinuity by definition
            if p.kind == "ramp" and (abs(t - p.t_start) < 1e-3 or abs(t - p.t_end) < 1e-3):
                continue  # kinks
            h = 1e-6
            a0, ad, _ = prescribed_alpha(p, t)
            a_m, _, _ = prescribed_alpha(p, t - h)
            a_p, _, _ = prescribed_alpha(p, t + h)
            assert abs(ad - (a_p - a_m) / (2 * h)) < 1e-6 * (1 + abs(ad)), (p.kind, t)
            _, _, add = prescribed_alpha(p, t)
            dd_fd = (a_p - 2 * a0 + a_m) / h**2
            assert abs(add - dd_fd) < 1e-4 * (1 + abs(dd_fd)), (p.kind, t)


def test_profile_rate_limit():
    """Stage 2: a command faster than rate_max is tracked at the limit."""
    import numpy as np

    from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
    from fwmav_sim.actuation.profile_limited import ProfileLimiter

    limiter = ProfileLimiter(rate_max=1.0, accel_max=50.0, kp=100.0, kd=30.0)
    cmd = AbdomenProfile(kind="sinusoid", amplitude=0.5, frequency=10.0)  # way too fast
    # integrate the limiter at fine dt
    state = np.zeros(2)
    dt = 1e-5
    peak_rate = 0.0
    for k in range(20000):
        state = state + limiter.rhs(state, k * dt, cmd) * dt
        peak_rate = max(peak_rate, abs(state[1]))
    assert peak_rate <= 1.0 + 1e-6
    # and it does track (nonzero response)
    assert np.abs(state[0]) > 1e-3


def test_torque_saturation_recorded():
    """Stage 3: commands beyond tau_max saturate and set the flag."""
    import numpy as np

    from fwmav_sim.actuation.torque_driven import TorqueActuator

    act = TorqueActuator(tau_max=1e-5, tau_cmd=lambda t: 1e-4 * np.sin(2 * np.pi * t))
    tau, sat = act.applied_torque(0.25, 0.0, 0.0)  # peak of command
    assert sat and abs(tau) == 1e-5
    tau, sat = act.applied_torque(0.0, 0.0, 0.0)  # zero command
    assert not sat and tau == 0.0
