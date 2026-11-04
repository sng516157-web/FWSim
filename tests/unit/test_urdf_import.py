"""URDF import spike tests (DR-004, post-M5 item 3)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.vehicle.urdf_import import UrdfImportError, import_urdf

_BASIC = """<?xml version="1.0"?>
<robot name="two_body">
  <link name="thorax">
    <inertial>
      <mass value="0.011"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="2e-6" ixy="0" ixz="0" iyy="2e-6" iyz="0" izz="2e-6"/>
    </inertial>
  </link>
  <link name="abdomen">
    <inertial>
      <mass value="0.004"/>
      <origin xyz="-0.015 0 0" rpy="0 0 0"/>
      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="abdomen_pitch" type="revolute">
    <parent link="thorax"/>
    <child link="abdomen"/>
    <origin xyz="-0.015 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.6" upper="0.6" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def test_basic_two_body_import(tmp_path):
    f = tmp_path / "robot.urdf"
    f.write_text(_BASIC, encoding="utf-8")
    v = import_urdf(f)
    assert v.name == "two_body"
    bodies = {b.name: b for b in v.bodies}
    assert bodies["thorax"].mass.value == pytest.approx(0.011)
    assert bodies["abdomen"].com_offset.value == [-0.015, 0.0, 0.0]
    assert bodies["abdomen"].inertia.value[1][1] == pytest.approx(1e-6)
    (j,) = v.joints
    assert j.name == "abdomen_pitch" and j.type == "revolute"
    assert j.parent_body == "thorax" and j.child_body == "abdomen"
    assert j.axis == [0.0, 1.0, 0.0]
    assert j.location.value == [-0.015, 0.0, 0.0]
    assert j.limits == (-0.6, 0.6)
    assert v.data_status.startswith("cad_import_spike")
    # provenance honesty: CAD-computed, labelled estimated with the file reference
    assert v.bodies[0].mass.provenance.source == "estimated"
    assert "robot.urdf" in v.bodies[0].mass.provenance.reference


def test_fixed_joint_merging(tmp_path):
    urdf = """<robot name="merged">
      <link name="base"><inertial>
        <mass value="0.010"/><origin xyz="0 0 0"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
      </inertial></link>
      <link name="payload"><inertial>
        <mass value="0.002"/><origin xyz="0.01 0 0"/>
        <inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/>
      </inertial></link>
      <link name="tail"><inertial>
        <mass value="0.001"/><origin xyz="-0.02 0 0"/>
        <inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/>
      </inertial></link>
      <joint name="rigid_mount" type="fixed">
        <parent link="base"/><child link="payload"/>
        <origin xyz="0.03 0 0"/>
      </joint>
      <joint name="tail_pitch" type="revolute">
        <parent link="payload"/><child link="tail"/>
        <origin xyz="0.01 0 0"/><axis xyz="0 1 0"/>
      </joint>
    </robot>"""
    f = tmp_path / "merged.urdf"
    f.write_text(urdf, encoding="utf-8")
    v = import_urdf(f)
    # payload folds into base; tail's joint re-parents to base with its origin
    # re-expressed: 0.03 (mount) + 0.01 (in payload frame) = 0.04
    assert {b.name for b in v.bodies} == {"base", "tail"}
    base = next(b for b in v.bodies if b.name == "base")
    m = 0.012
    com_x = 0.002 * (0.03 + 0.01) / m  # payload CoM in base frame at x=0.04
    assert base.mass.value == pytest.approx(m)
    assert base.com_offset.value[0] == pytest.approx(com_x)
    # inertia about the merged CoM: parallel axis of both parts
    i_base = 1e-6 + 0.010 * com_x**2
    i_pay = 1e-8 + 0.002 * (0.04 - com_x) ** 2
    assert base.inertia.value[1][1] == pytest.approx(i_base + i_pay, rel=1e-9)
    (j,) = v.joints
    assert j.parent_body == "base" and j.child_body == "tail"
    assert j.location.value[0] == pytest.approx(0.04)


def test_inertial_rpy_rotation(tmp_path):
    urdf = """<robot name="rot">
      <link name="base"><inertial>
        <mass value="0.001"/><origin xyz="0 0 0"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
      </inertial></link>
      <link name="rotor"><inertial>
        <mass value="0.001"/><origin xyz="0 0 0" rpy="0 1.5707963267948966 0"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="2e-6" iyz="0" izz="3e-6"/>
      </inertial></link>
      <joint name="j" type="revolute">
        <parent link="base"/><child link="rotor"/>
        <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
      </joint>
    </robot>"""
    f = tmp_path / "rot.urdf"
    f.write_text(urdf, encoding="utf-8")
    v = import_urdf(f)
    rotor = next(b for b in v.bodies if b.name == "rotor")
    # inertial frame pitched +90 deg about y: link x-axis sees inertia i_zz
    I_rot = np.asarray(rotor.inertia.value)
    np.testing.assert_allclose(np.diag(I_rot), [3e-6, 2e-6, 1e-6], atol=1e-15)


def test_reject_prismatic(tmp_path):
    urdf = """<robot name="prism">
      <link name="a"/><link name="b"/>
      <joint name="j" type="prismatic"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0"/><axis xyz="1 0 0"/></joint>
    </robot>"""
    f = tmp_path / "p.urdf"
    f.write_text(urdf, encoding="utf-8")
    with pytest.raises(UrdfImportError, match="prismatic"):
        import_urdf(f)


def test_reject_branch(tmp_path):
    urdf = """<robot name="branch">
      <link name="a"/><link name="b"/><link name="c"/>
      <joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>
      <joint name="j2" type="fixed"><parent link="a"/><child link="b"/></joint>
    </robot>"""
    f = tmp_path / "b.urdf"
    f.write_text(urdf, encoding="utf-8")
    with pytest.raises(UrdfImportError, match="two parents"):
        import_urdf(f)


def test_reject_non_axis_joint(tmp_path):
    urdf = """<robot name="na">
      <link name="a"><inertial><mass value="0.001"/><origin xyz="0 0 0"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial></link>
      <link name="b"><inertial><mass value="0.001"/><origin xyz="0 0 0"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial></link>
      <joint name="j" type="revolute"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0"/><axis xyz="0.7071 0.7071 0"/></joint>
    </robot>"""
    f = tmp_path / "na.urdf"
    f.write_text(urdf, encoding="utf-8")
    with pytest.raises(UrdfImportError, match="coordinate axis"):
        import_urdf(f)
