"""Aero coefficient registry tests (M6 Task 6)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.aero.registry import (
    available_sets,
    coefficient_set_info,
    resolve_coefficients,
)
from fwmav_sim.vehicle.schema import AeroModelSpec


def test_dickinson_set_matches_legacy_values():
    info = coefficient_set_info("dickinson1999_robofly")
    assert info["coefficients"]["cl_slope"] == pytest.approx(1.58)
    assert info["source"] == "literature"
    assert "Robofly" in info["regime"]


def test_poststall_set_evaluates_exactly():
    """The poststall registry set is the classical model, evaluated exactly."""
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.vehicle.schema import AeroModelSpec

    spec = AeroModelSpec(name="quasi_steady_blade_element",
                         coefficient_set="flatplate_poststall_classical")
    model = QuasiSteadyBladeElement(spec)
    for a_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        a = np.radians(a_deg)
        cl, cd = model.cl_cd(a)
        assert cl == pytest.approx(2.0 * np.sin(a) ** 2 * np.cos(a), abs=1e-12)
        assert cd == pytest.approx(2.0 * np.sin(a) ** 3, abs=1e-12)
    # odd CL / even CD folding (DR-003) applies to this form too
    cl_p, cd_p = model.cl_cd(np.radians(30.0))
    cl_m, cd_m = model.cl_cd(np.radians(-30.0))
    assert cl_m == pytest.approx(-cl_p, abs=1e-12)
    assert cd_m == pytest.approx(cd_p, abs=1e-12)


def test_exclusivity_enforced():
    from fwmav_sim.vehicle.schema import Parameter, Provenance

    prov = Provenance(source="estimated")
    inline = {"cl_0": Parameter(value=0.225, unit="1", provenance=prov)}
    with pytest.raises(ValueError, match="exactly one"):
        resolve_coefficients(AeroModelSpec(name="m", coefficients=inline,
                                           coefficient_set="dickinson1999_robofly"))
    with pytest.raises(ValueError, match="exactly one"):
        resolve_coefficients(AeroModelSpec(name="m", coefficients={}))


def test_named_set_builds_model_and_metadata():
    spec = AeroModelSpec(name="quasi_steady_blade_element",
                         coefficient_set="dickinson1999_robofly")
    model = QuasiSteadyBladeElement(spec)
    cl, cd = model.cl_cd(np.radians(45.0))
    assert cl == pytest.approx(0.225 + 1.58 * np.sin(np.radians(2.13 * 45 - 7.2)))
    assert model.set_info["regime"].startswith("Robofly")


def test_unknown_set_rejected():
    with pytest.raises(ValueError, match="unknown coefficient set"):
        coefficient_set_info("nonsense")
    assert "dickinson1999_robofly" in available_sets()
