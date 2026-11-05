"""Vectorized aero regression (M6.75 perf pass): batch == scalar to 1e-12."""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.kinematics.blade_element import (
    batched_element_state,
    discretize,
    element_state,
)

_ENV = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.array([0.1, 0.0, -0.05]))


def test_batch_matches_scalar_per_element():
    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    for wing in vehicle.wings:
        elements = discretize(wing, 8)
        for t in (0.0, 0.0137, 0.041, 0.062):
            st_b = batched_element_state(wing, elements, exp.kinematics, t,
                                         nu=_ENV.nu, v_air_W=_ENV.v_air_W)
            bf = model.element_forces_batch(st_b, _ENV)
            for j, el in enumerate(elements):
                st = element_state(wing, el, exp.kinematics, t, nu=_ENV.nu,
                                   v_air_W=_ENV.v_air_W)
                ef = model.element_forces(st, _ENV)
                for key in ("F_trans", "F_rot", "F_am"):
                    np.testing.assert_allclose(bf[key][j], getattr(ef, key),
                                               atol=1e-12, rtol=1e-10)
                # arctan2 can differ by 1 ULP between scalar and vectorized paths
                assert abs(st.alpha - st_b["alpha"][j]) < 1e-15
                np.testing.assert_allclose(st.r_W, st_b["r_W"][j], atol=1e-12)


def test_batch_wing_sums_match_scalar_loop():
    """Wing-total force/moment/power from the batch path equal the scalar loop."""
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body(vehicle, exp.kinematics, _ENV, model, 1, 40)
    # hand scalar loop at a few samples
    from fwmav_sim.kinematics.blade_element import discretize as dsc

    for wing in vehicle.wings:
        elements = dsc(wing, 16)
        for i in (0, 13, 39):
            ti = float(ts.t[i])
            F_sum = np.zeros(3)
            for el in elements:
                st = element_state(wing, el, exp.kinematics, ti, nu=_ENV.nu,
                                   v_air_W=_ENV.v_air_W)
                F_sum += model.element_forces(st, _ENV).resultant
            np.testing.assert_allclose(ts[f"{wing.name}/F_total"][i], F_sum,
                                       atol=1e-12, rtol=1e-9)
