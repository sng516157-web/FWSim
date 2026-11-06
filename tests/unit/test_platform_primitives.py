"""Mechanical tree editor: primitives + outline wings (M9 T2)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.ui.nodes.planform_canvas import PlanformOutline
from fwmav_sim.vehicle.platform_builder import PlatformBuilder


def test_box_primitive_inertia():
    pb = PlatformBuilder("prim_test")
    pb.add_body_primitive("thorax", "box",
                          {"lx": 0.04, "ly": 0.02, "lz": 0.02}, density=500.0)
    vd = pb.to_vehicle_dict()
    b = vd["bodies"][0]
    m = 500.0 * 0.04 * 0.02 * 0.02
    assert b["mass"]["value"] == pytest.approx(m)
    # box inertia: I_xx = m (ly² + lz²)/12
    assert b["inertia"]["value"][0][0] == pytest.approx(m * (0.02**2 + 0.02**2) / 12)


def test_cylinder_and_point_primitives():
    pb = PlatformBuilder("prim2")
    pb.add_body_primitive("rod", "cylinder", {"radius": 0.005, "length": 0.06},
                          mass=0.003)
    pb.add_body_primitive("payload", "point", {}, mass=0.002)
    vd = pb.to_vehicle()  # validates through the schema
    assert len(vd.bodies) == 2
    assert vd.bodies[1].inertia.value[0][0] < 1e-9  # point: negligible inertia


def test_point_needs_mass():
    pb = PlatformBuilder("p3")
    with pytest.raises(ValueError, match="explicit mass"):
        pb.add_body_primitive("p", "point", {}, density=1000.0)


def test_wing_from_outline_compiles(tmp_path):
    pb = PlatformBuilder("outline_test")
    pb.add_body_primitive("thorax", "point", {}, mass=0.02)
    outline = PlanformOutline([{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                               {"y": 0.15, "chord": 0.18, "le_offset": 0.06},
                               {"y": 0.3, "chord": 0.05, "le_offset": 0.12}])
    pb.add_wing_from_outline("wing_right", "right", outline)
    pb.add_wing_from_outline("wing_left", "left", outline)
    vd = pb.to_vehicle()
    assert vd.wings[0].planform == "custom"
    assert len(vd.wings[0].planform_stations) == 3
    out = pb.save_vehicle(tmp_path / "v.yaml")
    from fwmav_sim.experiment.config import load_vehicle

    vd2 = load_vehicle(out)
    assert vd2.wings[0].planform_stations[2].chord == pytest.approx(0.05)


def test_full_platform_from_primitives_runs():
    """The M9 headline: build a platform purely from primitives and run a
    fixed-body aero pass on it — no CAD, no YAML hand-editing."""
    import numpy as np

    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    pb = PlatformBuilder("primitives_only")
    pb.add_body_primitive("thorax", "box", {"lx": 0.04, "ly": 0.02, "lz": 0.02},
                          density=500.0)
    outline = PlanformOutline()  # default 0.3 m x 0.2 m rectangle
    pb.add_wing_from_outline("wing_right", "right", outline,
                             hinge_location=[0.0, 0.01, -0.005])
    pb.add_wing_from_outline("wing_left", "left", outline,
                             hinge_location=[0.0, -0.01, -0.005])
    vehicle = pb.to_vehicle()
    prov = {"source": "estimated", "uncertainty_or_range": "test placeholder"}
    p = lambda v, u: {"value": v, "unit": u, "provenance": prov}  # noqa: E731

    kin = PrescribedWingKinematics.model_validate({
        "flap_amplitude": p(1.0, "rad"), "flap_frequency": p(15.0, "Hz"),
        "pitch_amplitude": p(0.785, "rad"), "pitch_phase": p(3.14159, "rad"),
        "pitch_offset": p(0.0, "rad"), "stroke_plane_angle": p(0.0, "rad")})
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body(vehicle, {w.name: kin for w in vehicle.wings}, env, model,
                        n_wingbeats=1, steps_per_wingbeat=20)
    Fz = ts["wing_right/F_total"][:, 2]
    assert np.all(np.isfinite(Fz))
    assert np.abs(Fz).max() > 0.0  # nonzero aero forces on the custom outline
