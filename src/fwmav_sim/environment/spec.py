"""Build an Environment from the schema's EnvironmentSpec."""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import Environment
from fwmav_sim.vehicle.schema import EnvironmentSpec


def environment_from_spec(spec: EnvironmentSpec) -> Environment:
    return Environment(
        rho=spec.air_density.value,
        nu=spec.kinematic_viscosity.value,
        v_air_W=np.asarray(spec.wind.value, dtype=float),
    )
