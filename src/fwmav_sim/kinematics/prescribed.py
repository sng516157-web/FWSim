"""Prescribed wing kinematics (assumption A-004).

Frozen time-domain definitions (docs/conventions/frames.md §3), with w = 2 pi f:

    phi(t)     = phi_amp * cos(w t)                      stroke (flapping)
    theta_w(t) = theta_off + theta_amp * sin(w t + delta_theta) + Fourier terms
    psi_w(t)   = psi_amp * sin(w t + delta_psi)          deviation (optional)

Note (documented M0->M1 resolution): ``pitch_phase`` maps directly onto
``delta_theta`` of the frozen formula; positive delta_theta *advances* the
pitch profile relative to the stroke. The M1 stage file's comment "lag
relative to stroke" is superseded by the frozen formula.

All derivatives are computed analytically — no finite differences inside the
kinematics layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fwmav_sim.vehicle.schema import Parameter


class FourierTerm(BaseModel):
    """One term a_n * sin(2 pi n f t + delta_n) of an extended pitch profile."""

    model_config = ConfigDict(extra="forbid")

    amplitude: float  # rad
    harmonic: int = Field(ge=1)
    phase: float = 0.0  # rad


class PrescribedWingKinematics(BaseModel):
    """Prescribed sinusoidal (+ optional Fourier) wing kinematics."""

    model_config = ConfigDict(extra="forbid")

    flap_amplitude: Parameter  # phi_amp [rad]
    flap_frequency: Parameter  # f [Hz]
    pitch_amplitude: Parameter  # theta_amp [rad]
    pitch_phase: Parameter  # delta_theta [rad]; positive = advance
    pitch_offset: Parameter  # theta_off [rad]
    deviation_amplitude: Parameter | None = None  # psi_amp [rad]
    deviation_phase: Parameter | None = None  # delta_psi [rad]
    stroke_plane_angle: Parameter  # beta [rad]
    pitch_fourier: list[FourierTerm] = Field(default_factory=list)


@dataclass(frozen=True)
class WingKinematicState:
    """Wing angles and their first two time derivatives at one instant."""

    t: float
    phi: float
    phi_dot: float
    phi_ddot: float
    theta_w: float
    theta_w_dot: float
    theta_w_ddot: float
    psi_w: float
    psi_w_dot: float
    psi_w_ddot: float


def wing_state(kin: PrescribedWingKinematics, t: float) -> WingKinematicState:
    """Analytic wing kinematic state at time t (frames.md §3)."""
    phi_amp = kin.flap_amplitude.value
    w = 2.0 * np.pi * kin.flap_frequency.value
    theta_amp = kin.pitch_amplitude.value
    delta_theta = kin.pitch_phase.value
    theta_off = kin.pitch_offset.value

    phi = phi_amp * np.cos(w * t)
    phi_dot = -phi_amp * w * np.sin(w * t)
    phi_ddot = -phi_amp * w * w * np.cos(w * t)

    theta_w = theta_off + theta_amp * np.sin(w * t + delta_theta)
    theta_w_dot = theta_amp * w * np.cos(w * t + delta_theta)
    theta_w_ddot = -theta_amp * w * w * np.sin(w * t + delta_theta)
    for term in kin.pitch_fourier:
        nw = term.harmonic * w
        theta_w += term.amplitude * np.sin(nw * t + term.phase)
        theta_w_dot += term.amplitude * nw * np.cos(nw * t + term.phase)
        theta_w_ddot -= term.amplitude * nw * nw * np.sin(nw * t + term.phase)

    if kin.deviation_amplitude is not None:
        psi_amp = kin.deviation_amplitude.value
        delta_psi = kin.deviation_phase.value if kin.deviation_phase is not None else 0.0
        psi_w = psi_amp * np.sin(w * t + delta_psi)
        psi_w_dot = psi_amp * w * np.cos(w * t + delta_psi)
        psi_w_ddot = -psi_amp * w * w * np.sin(w * t + delta_psi)
    else:
        psi_w = psi_w_dot = psi_w_ddot = 0.0

    return WingKinematicState(
        t=t,
        phi=phi, phi_dot=phi_dot, phi_ddot=phi_ddot,
        theta_w=theta_w, theta_w_dot=theta_w_dot, theta_w_ddot=theta_w_ddot,
        psi_w=psi_w, psi_w_dot=psi_w_dot, psi_w_ddot=psi_w_ddot,
    )
