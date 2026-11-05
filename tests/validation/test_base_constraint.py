"""Unified base constraint tests (DR-007, M6.75).

- alias resolution: kinds imply constraints; base blocks override (recorded)
- clamped plant reproduces the fast rig runner's forces
- release transient: static until release, continuous state at release
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.experiment.config import (
    BaseConstraintSpec,
    ExperimentDefinition,
    effective_base,
)


def _exp(kind, base=None):
    fields = {"name": "t", "kind": kind, "base": base}
    return ExperimentDefinition.model_validate({k: v for k, v in fields.items() if v is not None})


def test_kind_implied_constraints():
    assert effective_base(_exp("fixed_body"))[0] == "fixed"
    assert effective_base(_exp("free_body"))[0] == "free"
    assert effective_base(_exp("suspended_pendulum"))[0] == "pivot"


def test_base_override_recorded():
    base = BaseConstraintSpec(mode="free")
    mode, warnings = effective_base(_exp("fixed_body", base))
    assert mode == "free"
    assert warnings and "overrides" in warnings[0]


def test_release_time_only_with_free():
    with pytest.raises(ValueError, match="release_time"):
        effective_base(_exp("fixed_body", BaseConstraintSpec(mode="fixed", release_time=0.1)))


def test_pivot_requires_point():
    with pytest.raises(ValueError, match="pivot"):
        effective_base(_exp("free_body", BaseConstraintSpec(mode="pivot")))


def test_clamped_plant_matches_rig_runner():
    """free_body plant clamped (base fixed) == fast fixed_body forces."""
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.experiment.config import load_experiment, load_vehicle
    from fwmav_sim.simulation.fixed_body import run_fixed_body
    from fwmav_sim.simulation.free_body import AbdomenControl, FreeBodyPlant

    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    model = QuasiSteadyBladeElement(vehicle.aero)
    kin_map = {w.name: exp.kinematics for w in vehicle.wings}

    ts = run_fixed_body(vehicle, exp.kinematics, env, model, 2, 50)

    plant = FreeBodyPlant(vehicle, kin_map, env, model,
                          control=AbdomenControl(mode="locked"),
                          base_mode="fixed")
    from fwmav_sim.simulation.integrator import pack_state

    y0 = pack_state([0, 0, 0], [1, 0, 0, 0], [0.0, 0.0],
                    [0, 0, 0], [0, 0, 0], [0.0, 0.0])
    for i, ti in enumerate(ts.t[::7]):
        plant.rhs(float(ti), y0)
        for w in vehicle.wings:
            rec = plant.last_F_wings[w.name]
            F_plant = rec["F_trans"] + rec["F_rot"] + rec["F_am"]
            F_rig = ts[f"{w.name}/F_total"][i * 7]
            np.testing.assert_allclose(F_plant, F_rig, atol=1e-9)


def test_release_transient_continuity():
    """Clamped until release_time: the state holds exactly at the initial
    pose until release, then the body accelerates from rest."""
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.experiment.config import load_experiment, load_vehicle
    from fwmav_sim.simulation.free_body import AbdomenControl, FreeBodyPlant
    from fwmav_sim.simulation.integrator import integrate_stepped, pack_state

    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    model = QuasiSteadyBladeElement(vehicle.aero)
    kin_map = {w.name: exp.kinematics for w in vehicle.wings}
    f = exp.kinematics.flap_frequency.value
    T_rel = 1.0 / f

    plant = FreeBodyPlant(vehicle, kin_map, env, model,
                          control=AbdomenControl(mode="locked"),
                          release_time=T_rel)
    y0 = pack_state([0, 0, 0], [1, 0, 0, 0], [0.0, 0.0],
                    [0, 0, 0], [0, 0, 0], [0.0, 0.0])
    t, Y = integrate_stepped(plant.rhs, y0, 3.0 / f, rtol=1e-9, atol=1e-12,
                             sample_dt=0.005)
    pre = t < T_rel - 1e-9
    # while clamped: position and attitude frozen at the initial state
    np.testing.assert_allclose(Y[0:3, pre], 0.0, atol=1e-12)
    quat0 = np.array([[1.0], [0.0], [0.0], [0.0]])
    assert np.abs(Y[3:7, pre] - quat0).max() < 1e-12
    # velocity at the release instant is continuous (zero on both sides)
    i_rel = int(np.argmin(np.abs(t - T_rel)))
    np.testing.assert_allclose(Y[7:10, i_rel], 0.0, atol=1e-9)
    # after release the body moves (gravity wins over hover under-trim)
    assert np.linalg.norm(Y[0:3, -1]) > 1e-6
    assert t[-1] > 2.5 / f
