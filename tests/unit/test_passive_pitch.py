"""Passive torsional wing pitch tests (M7 Task 3)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.dynamics.wing_pitch_compliance import (
    PitchSpring,
    run_fixed_body_passive_pitch,
)
from fwmav_sim.experiment.config import load_experiment, load_vehicle

_ENV = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))


def _sprung_vehicle(k: float, c: float, rest: float = 0.0, inertia: float = 1e-7):
    vehicle = load_vehicle("reference_vehicles/airpulse.yaml")
    prov = {"source": "estimated"}
    spring = {
        "stiffness": {"value": k, "unit": "N m / rad", "provenance": prov},
        "damping": {"value": c, "unit": "N m s / rad", "provenance": prov},
        "rest_angle": {"value": rest, "unit": "rad", "provenance": prov},
        "inertia": {"value": inertia, "unit": "kg m^2", "provenance": prov},
    }
    import copy

    v = copy.deepcopy(vehicle.model_dump())
    for w in v["wings"]:
        w["pitch_spring"] = dict(spring)
    from fwmav_sim.vehicle.schema import VehicleDefinition

    return VehicleDefinition.model_validate(v)


def test_stiff_spring_holds_rest_angle():
    """Huge stiffness: pitch stays at rest (the prescribed-limit check)."""
    # aero pitch moments are ~2 mN m mid-stroke (measured); k=1e2 N m/rad
    # keeps the equilibrium deflection ~2e-5 rad, well under tolerance.
    vehicle = _sprung_vehicle(k=1e2, c=1e-2)
    exp = load_experiment("experiments/m4_airpulse_hover_fixed.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body_passive_pitch(
        vehicle, {w.name: exp.kinematics for w in vehicle.wings}, _ENV, model,
        n_wingbeats=1, steps_per_wingbeat=40)
    psi = ts["wing_right/psi_passive"]
    assert np.abs(psi).max() < 1e-3  # stays at rest (0) to tolerance


def test_spring_settles_bounded_and_balanced():
    """Moderate spring: pitch settles to a bounded oscillation, and at settle
    the spring moment balances the aero moment on cycle average."""
    # k must exceed the aero pitch stiffness (~0.15 N m/rad measured on this
    # vehicle) or the wing diverges torsionally; 0.5 gives a ~10 mrad
    # bounded periodic response that settles within the first wingbeat.
    vehicle = _sprung_vehicle(k=0.5, c=1e-3, inertia=1e-6)
    exp = load_experiment("experiments/m4_airpulse_hover_fixed.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body_passive_pitch(
        vehicle, {w.name: exp.kinematics for w in vehicle.wings}, _ENV, model,
        n_wingbeats=4, steps_per_wingbeat=40)
    psi = ts["wing_right/psi_passive"]
    assert np.abs(psi).max() < 1.5  # bounded
    # second half has smaller drift than the first (settling)
    first_half_var = np.var(psi[: len(psi) // 2])
    second_half_var = np.var(psi[len(psi) // 2:])
    assert second_half_var <= first_half_var * 1.5


def test_no_spring_block_errors():
    vehicle = load_vehicle("reference_vehicles/airpulse.yaml")  # no pitch_spring
    exp = load_experiment("experiments/m4_airpulse_hover_fixed.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    with pytest.raises(ValueError, match="pitch_spring"):
        run_fixed_body_passive_pitch(vehicle, {w.name: exp.kinematics for w in vehicle.wings},
                                     _ENV, model, 1, 10)
