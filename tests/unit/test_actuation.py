"""§11.1 actuator tests.

Real (M2): prescribed abdomen profile derivatives. Stubs (M3): profile rate
limiting and torque saturation recording (actuator stages 2-3).
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.skip(reason="stub — implemented in M3 (actuator stage 2)")
def test_profile_rate_limit():
    raise NotImplementedError


@pytest.mark.skip(reason="stub — implemented in M3 (actuator stage 3)")
def test_torque_saturation_recorded():
    raise NotImplementedError
