"""Prescribed abdomen angle profiles (M2 Task 6; brief §6.4 stage 1).

The abdomen angle alpha_abd(t) is prescribed, not torque-driven — the actuator
is assumed ideal in this mode (assumption: brief §5.6 mode 1). Profiles have
analytic first/second derivatives. Optional clamping to the joint limits from
JointSpec; clamped samples report zero rate (the profile is saturated).

Commanded vs actual: in prescribed mode the actual angle equals the (clamped)
command by construction; both are recorded in the run record so later
actuator stages can be compared against the same channel layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AbdomenProfile:
    """Prescribed alpha_abd(t). Angles in rad, frequency in Hz, times in s.

    kind:
      "step"     — offset until t_start, then offset + amplitude (rates zero;
                   discontinuous at t_start by definition of an ideal step)
      "ramp"     — linear from offset to offset + amplitude over
                   [t_start, t_end], constant outside
      "sinusoid" — offset + amplitude * sin(2 pi frequency t + phase)
    """

    kind: Literal["step", "ramp", "sinusoid"]
    amplitude: float = 0.0  # rad
    offset: float = 0.0  # rad
    frequency: float = 0.0  # Hz (sinusoid only)
    phase: float = 0.0  # rad (sinusoid only)
    t_start: float = 0.0  # s (step/ramp)
    t_end: float = 1.0  # s (ramp only)


def prescribed_alpha(
    profile: AbdomenProfile, t: float, limits: tuple[float, float] | None = None
) -> tuple[float, float, float]:
    """Return (alpha, alpha_dot, alpha_ddot) at time t, optionally clamped."""
    p = profile
    if p.kind == "step":
        a = p.offset + (p.amplitude if t >= p.t_start else 0.0)
        ad = add = 0.0
    elif p.kind == "ramp":
        slope = p.amplitude / (p.t_end - p.t_start)
        if t <= p.t_start:
            a, ad, add = p.offset, 0.0, 0.0
        elif t >= p.t_end:
            a, ad, add = p.offset + p.amplitude, 0.0, 0.0
        else:
            a, ad, add = p.offset + slope * (t - p.t_start), slope, 0.0
    elif p.kind == "sinusoid":
        w = 2.0 * np.pi * p.frequency
        a = p.offset + p.amplitude * np.sin(w * t + p.phase)
        ad = p.amplitude * w * np.cos(w * t + p.phase)
        add = -p.amplitude * w * w * np.sin(w * t + p.phase)
    else:
        raise ValueError(f"unknown profile kind {p.kind!r}")

    if limits is not None:
        lo, hi = limits
        if a < lo:
            return lo, 0.0, 0.0
        if a > hi:
            return hi, 0.0, 0.0
    return a, ad, add
