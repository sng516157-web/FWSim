"""GL frame adapter tests (M9 T5): the adapter must produce geometry
consistent with the MeshCat path (same math, two renderers)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.experiment.config import kinematics_map, load_experiment, load_vehicle
from fwmav_sim.experiment.record import load_record
from fwmav_sim.viz.gl_frames import frame_geometry, wing_mesh_view
from fwmav_sim.viz.meshcat_scene import NED_TO_VIEW, _FreeBodyReplayCtx


@pytest.fixture(scope="module")
def ctx():
    """A replay context from a real (short) fixed-body-style record."""
    import subprocess
    from pathlib import Path

    # reuse an existing free-body output if present, else skip
    outs = sorted(Path("outputs").glob("*m3_free_body_hover"))
    if not outs:
        pytest.skip("no free-body record available to adapt")
    ts, _ = load_record(outs[-1])
    vehicle = load_vehicle(outs[-1] / "vehicle.yaml")
    exp = load_experiment(outs[-1] / "config.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    model = QuasiSteadyBladeElement(vehicle.aero)
    kin_map = kinematics_map(exp, vehicle)
    return _FreeBodyReplayCtx(ts=ts, vehicle=vehicle, env=env, model=model,
                              kin_map=kin_map, art=None, state=None,
                              force_scale=5.0)


def test_frame_geometry_shapes(ctx):
    meshes, arrows = frame_geometry(ctx, 0)
    assert len(meshes) == len(ctx.vehicle.wings)
    verts, faces = meshes[0]
    assert verts.shape[1] == 3 and faces.shape[1] == 3
    assert np.all(np.isfinite(verts))
    for o, v in arrows:
        assert o.shape == (3,) and v.shape == (3,)


def test_wing_mesh_honours_custom_outline():
    """A swept custom outline must shift the tip chordwise in the mesh."""
    from fwmav_sim.vehicle.schema import VehicleDefinition

    prov = {"source": "estimated"}
    p = lambda v, u: {"value": v, "unit": u, "provenance": prov}  # noqa: E731
    wing = {
        "name": "w", "side": "right", "span": p(0.3, "m"),
        "root_chord": p(0.2, "m"), "tip_chord": p(0.2, "m"),
        "planform": "custom",
        "planform_stations": [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                              {"y": 0.3, "chord": 0.2, "le_offset": 0.15}],
        "hinge_location": p([0, 0, 0], "m"), "pitch_axis_fraction": p(0.25, "1"),
    }
    vd = VehicleDefinition.model_validate({
        "name": "t", "data_status": "estimated",
        "bodies": [{"name": "b", "mass": p(0.02, "kg"), "com_offset": p([0, 0, 0], "m"),
                    "inertia": p([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]], "kg m^2")}],
        "joints": [], "wings": [wing],
        "blade_elements": {"n_elements": 8, "discretization": "equal_span"},
        "aero": {"name": "none", "coefficients": {}}})
    verts, _ = wing_mesh_view(vd.wings[0], np.eye(3), np.zeros(3), 8)
    # verts alternate TE/LE per station; the LE x at the tip should exceed
    # the LE x at the root (swept forward in +x toward LE)
    le_root_x = verts[1, 0]
    le_tip = verts[-1, 0]
    assert le_tip > le_root_x
