"""Platform builder tests (M8 Task 3)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.vehicle.platform_builder import PlatformBuilder


def _minimal_platform():
    pb = PlatformBuilder("test_platform")
    pb.add_body("thorax", mass=0.02)
    pb.add_wing("wing_right", side="right", span=0.3, root_chord=0.18)
    pb.add_wing("wing_left", side="left", span=0.3, root_chord=0.18)
    return pb


def test_build_and_validate():
    vd = _minimal_platform().to_vehicle()
    assert vd.name == "test_platform"
    assert len(vd.bodies) == 1 and len(vd.wings) == 2
    assert vd.wings[0].span.value == 0.3


def test_duplicate_names_rejected():
    pb = _minimal_platform()
    with pytest.raises(ValueError, match="duplicate body"):
        pb.add_body("thorax", mass=0.02)
    with pytest.raises(ValueError, match="duplicate wing"):
        pb.add_wing("wing_right", side="right", span=0.3, root_chord=0.18)


def test_joint_requires_existing_bodies():
    pb = _minimal_platform()
    with pytest.raises(ValueError, match="existing bodies"):
        pb.add_joint("j", "revolute", "thorax", "ghost", [0, 1, 0])


def test_tree_roundtrip_through_yaml(tmp_path):
    pb = _minimal_platform()
    pb.add_body("abdomen", mass=0.005, com_offset=[-0.02, 0.0, 0.0])
    pb.add_joint("abd_pitch", "revolute", "thorax", "abdomen", [0, 1, 0],
                 origin=[-0.01, 0.0, 0.0], limits=[-0.6, 0.6])
    out = pb.save_vehicle(tmp_path / "platform.yaml")
    from fwmav_sim.experiment.config import load_vehicle

    vd = load_vehicle(out)
    assert len(vd.joints) == 1
    assert vd.joints[0].parent_body == "thorax"
    assert vd.joints[0].child_body == "abdomen"
    np.testing.assert_allclose(vd.joints[0].axis, [0, 1, 0])


def test_from_vehicle_roundtrip():
    pb = PlatformBuilder.from_vehicle("reference_vehicles/airpulse.yaml")
    vd = pb.to_vehicle()
    assert vd.name == "airpulse"
    assert len(vd.wings) == 2


def test_from_urdf_import(tmp_path):
    urdf = tmp_path / "mini.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="mini">
  <link name="thorax">
    <inertial><mass value="0.02"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
</robot>""")
    pb = PlatformBuilder.from_urdf(urdf)
    pb.add_wing("wing_right", side="right", span=0.3, root_chord=0.18)
    pb.add_wing("wing_left", side="left", span=0.3, root_chord=0.18)
    vd = pb.to_vehicle()
    assert vd.bodies[0].mass.value == pytest.approx(0.02)
    assert len(vd.wings) == 2
