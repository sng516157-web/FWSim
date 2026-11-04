"""Actuator stage 3 (brief §6.4): torque-driven abdomen joint.

The abdomen joint is a free dynamic DoF driven by an actuator torque command
with saturation, plus joint damping and compliant joint-limit stops
(penalty stops, documented: parameters k_stop/c_stop are modelling choices,
not hardware data).

    tau_applied = clip(tau_cmd(t), +/- tau_max)          (saturation recorded)
                - c_a * alpha_dot                        (joint damping)
                + stop torque when beyond the limits

Stage 4 (servo dynamics + power estimates) is deferred: selecting it raises
NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TorqueActuator:
    """Saturated torque source with joint damping and compliant stops."""

    tau_max: float  # N m
    c_a: float = 0.0  # joint damping, N m s / rad
    limits: tuple[float, float] | None = None  # rad
    k_stop: float = 1e-3  # N m / rad (compliant stop stiffness; modelling choice)
    c_stop: float = 1e-4  # N m s / rad (stop damping; modelling choice)
    tau_cmd: Callable[[float], float] = lambda t: 0.0  # commanded torque

    #: (applied torque, saturated flag) at time t for joint state (alpha, alpha_dot)
    def applied_torque(self, t: float, alpha: float, alpha_dot: float) -> tuple[float, bool]:
        tau_c = float(self.tau_cmd(t))
        tau = float(np.clip(tau_c, -self.tau_max, self.tau_max))
        saturated = tau != tau_c
        tau -= self.c_a * alpha_dot
        if self.limits is not None:
            lo, hi = self.limits
            if alpha > hi:
                tau -= self.k_stop * (alpha - hi) + self.c_stop * max(alpha_dot, 0.0)
            elif alpha < lo:
                tau -= self.k_stop * (alpha - lo) + self.c_stop * min(alpha_dot, 0.0)
        return tau, saturated
