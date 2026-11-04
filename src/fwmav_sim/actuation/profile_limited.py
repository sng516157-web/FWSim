"""Actuator stage 2 (brief §6.4): rate/acceleration-limited abdomen profile.

A cascaded limiter: the commanded profile alpha_cmd(t) is tracked by a
second-order filter whose rate and acceleration are clamped to the actuator's
limits:

    alpha_dot_des = clamp(Kp (alpha_cmd - alpha), +/- rate_max)
    alpha_ddot    = clamp(Kd (alpha_dot_des - alpha_dot), +/- accel_max)

The limiter is a 2-state dynamical system integrated *inside* the plant (its
state is appended to the integrator state by the free-body runner). The
abdomen joint follows the limiter output as a prescribed motion (the joint
reaction torque is recorded by the plant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha


@dataclass(frozen=True)
class ProfileLimiter:
    """Cascaded rate/acceleration limiter for a commanded abdomen profile."""

    rate_max: float  # rad / s
    accel_max: float  # rad / s^2
    kp: float = 50.0  # 1/s
    kd: float = 20.0  # 1/s

    def rhs(self, state: np.ndarray, t: float, command: AbdomenProfile) -> np.ndarray:
        """state = [alpha, alpha_dot]; returns [alpha_dot, alpha_ddot]."""
        alpha, alpha_dot = state
        alpha_cmd = prescribed_alpha(command, t)[0]
        rate_des = np.clip(self.kp * (alpha_cmd - alpha), -self.rate_max, self.rate_max)
        alpha_ddot = np.clip(self.kd * (rate_des - alpha_dot), -self.accel_max, self.accel_max)
        return np.array([alpha_dot, alpha_ddot])
