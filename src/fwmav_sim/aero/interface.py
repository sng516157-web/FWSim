"""Aerodynamic model interface (M1 Task 4).

Every aerodynamic model consumes the single ``ElementState`` representation
(brief §5.2) plus an ``Environment``, and returns the three force components
**separately** — translational, rotational, added mass — never only a
resultant (brief §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fwmav_sim.kinematics.blade_element import ElementState


@dataclass
class Environment:
    """Air properties at an element (still air or uniform wind in M1)."""

    rho: float  # air density, kg/m^3
    nu: float  # kinematic viscosity, m^2/s
    v_air_W: np.ndarray  # (3,) air velocity in the world frame, m/s


@dataclass
class ElementForces:
    """Force components on one blade element, world frame, newtons."""

    F_trans: np.ndarray  # (3,) translational (circulatory + profile drag)
    F_rot: np.ndarray  # (3,) rotational circulation
    F_am: np.ndarray  # (3,) added mass
    F_unsteady: np.ndarray | None = None  # optional reduced-order terms (M6, A-002)

    @property
    def resultant(self) -> np.ndarray:
        out = self.F_trans + self.F_rot + self.F_am
        if self.F_unsteady is not None:
            out = out + self.F_unsteady
        return out


class AerodynamicModel(Protocol):
    """Brief §5.1 modular aero interface."""

    def element_forces(self, state: ElementState, env: Environment) -> ElementForces: ...
