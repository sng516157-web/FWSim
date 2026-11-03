"""§11.2 free-body gravity moment about the instantaneous CoM (M2 Task 2).

Gravity has no resultant moment about the instantaneous CoM (brief §6.3) —
pinned to machine precision over random configurations.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.dynamics.mass_properties import gravity_moment_about_com
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}


def _vehicle() -> VehicleDefinition:
    return VehicleDefinition.model_validate({
        "name": "grav_moment_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": 0.011, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[2e-6, 0, 0], [0, 2e-6, 0], [0, 0, 2e-6]],
                         "unit": "kg m^2", "provenance": _PROV}},
            {"name": "abdomen",
             "mass": {"value": 0.004, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [-0.03, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[5e-7, 0, 0], [0, 5e-7, 0], [0, 0, 5e-7]],
                         "unit": "kg m^2", "provenance": _PROV}},
        ],
        "joints": [{
            "name": "abdomen_pitch", "type": "revolute",
            "parent_body": "thorax", "child_body": "abdomen",
            "axis": [0.0, 1.0, 0.0],
            "location": {"value": [-0.015, 0.0, 0.0], "unit": "m", "provenance": _PROV},
        }],
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    })


def test_gravity_moment_about_com_zero():
    v = _vehicle()
    rng = np.random.default_rng(5)
    scale = 0.015 * 9.81 * 0.03  # m g l order of magnitude
    for _ in range(100):
        q = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1),
                      rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4),
                      rng.uniform(-np.pi, np.pi), rng.uniform(-0.6, 0.6)])
        moment = gravity_moment_about_com(q, v)
        assert np.all(np.abs(moment) < 1e-12 * scale), moment
