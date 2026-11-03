"""Still air (brief §4.1)."""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import Environment


def still_air(rho: float = 1.225, nu: float = 1.460e-5) -> Environment:
    """Still air. Defaults: ISA sea-level density/viscosity (literature)."""
    return Environment(rho=rho, nu=nu, v_air_W=np.zeros(3))
