"""Uniform steady wind (brief §4.1; assumption A-006)."""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import Environment


def uniform_wind(v_air_W: np.ndarray, rho: float = 1.225, nu: float = 1.460e-5) -> Environment:
    """Uniform steady wind; velocity expressed in the world frame W."""
    return Environment(rho=rho, nu=nu, v_air_W=np.asarray(v_air_W, dtype=float))
