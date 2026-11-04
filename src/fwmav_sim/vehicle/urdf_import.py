"""URDF -> VehicleDefinition import (DR-004 spike, post-M5 item 3).

Scope of the spike (explicitly bounded):

- Links become bodies: ``<inertial>`` mass, origin (CoM offset) and inertia
  tensor map directly onto BodySpec. Inertia is rotated from the inertial
  frame into the link frame via the origin's rpy.
- Revolute joints about a coordinate axis become JointSpec entries; the joint
  origin (xyz of the joint ``<origin>`` in the parent link frame) becomes the
  joint location. Continuous joints are treated as revolute without limits.
- Fixed joints are MERGED: the child subtree's mass is folded into the parent
  body (parallel-axis theorem), because the schema's chain is serial with one
  body per joint. Branches are rejected.
- Prismatic/planar/floating joints and multi-DOF chains are rejected with a
  clear error (schema MVP topology).

Frame convention: URDF carries no global convention; the importer assumes the
URDF frames already match ours (link frame = body frame, +x forward, +z down)
or that the caller pre-rotated at export. This is documented in the generated
YAML header comment. Wing planforms do NOT exist in URDF — wings are authored
manually in the vehicle YAML after import (the spike imports mass properties
and the joint tree only).

Provenance: values are CAD-computed, not physically measured, so they are
labelled ``estimated`` with the URDF file as reference (brief §5.8 honesty).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fwmav_sim.core.transforms import R_W_B
from fwmav_sim.vehicle.schema import VehicleDefinition


def _parse_origin(elem) -> tuple[np.ndarray, np.ndarray]:
    """(xyz, rpy) of a URDF <origin>; defaults to identity."""
    if elem is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in elem.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in elem.get("rpy", "0 0 0").split()])
    return xyz, rpy


def _parallel_axis(mass: float, com: np.ndarray, I_com: np.ndarray,
                   point: np.ndarray) -> np.ndarray:
    """Inertia about `point` given inertia about `com` (same orientation)."""
    d = np.asarray(com) - np.asarray(point)
    return I_com + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


class UrdfImportError(ValueError):
    """Unsupported or invalid URDF content (spike scope limits)."""


def import_urdf(path: str | Path, name: str | None = None) -> VehicleDefinition:
    """Parse a URDF file into a VehicleDefinition (bodies + joints only)."""
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise UrdfImportError(f"expected <robot>, got <{root.tag}>")

    links: dict[str, dict] = {}
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            # massless frame carrier (allowed by the schema via massless: true)
            links[link.get("name")] = {"mass": 0.0, "com": np.zeros(3),
                                       "I": np.zeros((3, 3)), "massless": True}
            continue
        mass = float(inertial.find("mass").get("value"))
        xyz, rpy = _parse_origin(inertial.find("origin"))
        ie = inertial.find("inertia")
        I_ief = np.array([
            [float(ie.get("ixx")), float(ie.get("ixy")), float(ie.get("ixz"))],
            [float(ie.get("ixy")), float(ie.get("iyy")), float(ie.get("iyz"))],
            [float(ie.get("ixz")), float(ie.get("iyz")), float(ie.get("izz"))],
        ])
        # URDF inertia is expressed in the INERTIAL frame (origin at the CoM,
        # rotated by rpy relative to the link frame): rotate into link coords.
        R_L_I = R_W_B(rpy[2], rpy[1], rpy[0])  # URDF rpy = ZYX extrinsic == R_W_B
        links[link.get("name")] = {"mass": mass, "com": xyz,
                                   "I": R_L_I @ I_ief @ R_L_I.T, "massless": False}

    joints = []
    children: dict[str, str] = {}
    for jt in root.iter("joint"):
        jtype = jt.get("type")
        if jtype not in ("revolute", "continuous", "fixed"):
            raise UrdfImportError(
                f"joint {jt.get('name')!r}: type {jtype!r} unsupported (spike scope)")
        parent = jt.find("parent").get("link")
        child = jt.find("child").get("link")
        if child in children:
            raise UrdfImportError(f"link {child!r} has two parents (branches unsupported)")
        children[child] = parent
        xyz, rpy = _parse_origin(jt.find("origin"))
        if np.any(np.abs(rpy) > 1e-12):
            raise UrdfImportError(
                f"joint {jt.get('name')!r}: rotated joint origins (rpy != 0) are not "
                "supported by the spike; align frames at CAD export")
        axis = np.array([0.0, 0.0, 0.0])
        if jtype != "fixed":
            ax = jt.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None
                                                else "1 0 0").split()])
            if not np.isclose(abs(axis).sum(), 1.0, atol=1e-9):
                raise UrdfImportError(
                    f"joint {jt.get('name')!r}: axis {axis} is not a coordinate axis "
                    "(ArticulatedModel restriction)")
        limits = None
        if jtype == "revolute":
            lim = jt.find("limit")
            if lim is not None:
                limits = (float(lim.get("lower")), float(lim.get("upper")))
        joints.append({"name": jt.get("name"), "type": jtype, "parent": parent,
                       "child": child, "axis": axis, "origin": xyz,
                       "limits": limits})

    # fixed-joint merging: fold each fixed child into its (moved) ancestor.
    # Link frames of merged-away links collapse onto the surviving ancestor;
    # joint origins of descendants are re-expressed by translation (rotated
    # origins were rejected above, so translation is exact).
    merged = {name: dict(spec) for name, spec in links.items()}
    reparent: dict[str, tuple[str, np.ndarray]] = {}  # link -> (survivor, offset)

    def survivor(link: str) -> tuple[str, np.ndarray]:
        if link not in reparent:
            return link, np.zeros(3)
        parent, off = reparent[link]
        root_link, root_off = survivor(parent)
        return root_link, root_off + off

    def depth(link: str) -> int:
        d, cur = 0, link
        while cur in children:
            cur = children[cur]
            d += 1
        return d

    # child-first so nested fixed chains (A -fixed-> B -fixed-> C) fold C into
    # B before B folds into A
    fixed = sorted((j for j in joints if j["type"] == "fixed"),
                   key=lambda j: -depth(j["child"]))
    for j in fixed:
        parent, _ = survivor(j["parent"])
        child = j["child"]
        cs, ps = merged[child], merged[parent]
        com_child_in_parent = j["origin"] + cs["com"]
        m_new = ps["mass"] + cs["mass"]
        if ps["massless"] and cs["massless"]:
            pass  # both massless: nothing to fold
        elif m_new > 0:
            com_new = (ps["mass"] * ps["com"] + cs["mass"] * com_child_in_parent) / m_new
            I_new = _parallel_axis(ps["mass"], ps["com"], ps["I"], com_new) \
                + _parallel_axis(cs["mass"], com_child_in_parent, cs["I"], com_new)
            merged[parent] = {"mass": m_new, "com": com_new, "I": I_new,
                              "massless": False}
        reparent[child] = (parent, np.asarray(j["origin"], dtype=float))
        del merged[child]

    active_joints = []
    for j in joints:
        if j["type"] == "fixed":
            continue
        parent, off = survivor(j["parent"])
        j["parent"] = parent
        j["origin"] = np.asarray(j["origin"], dtype=float) + off
        active_joints.append(j)

    prov = {"source": "estimated",
            "reference": f"CAD export via URDF: {path.name} (DR-004 import spike)",
            "uncertainty_or_range": "CAD solid-model mass properties; material "
                                    "density fidelity not verified against hardware"}
    bodies = [{
        "name": bname,
        "mass": {"value": spec["mass"], "unit": "kg", "provenance": dict(prov)},
        "com_offset": {"value": [float(x) for x in spec["com"]], "unit": "m",
                       "provenance": dict(prov)},
        "inertia": {"value": [[float(v) for v in row] for row in spec["I"]],
                    "unit": "kg m^2", "provenance": dict(prov)},
        **({"massless": True} if spec["massless"] else {}),
    } for bname, spec in merged.items()]

    # our schema: revolute only (continuous -> limit-free revolute)
    joint_specs = [{
        "name": j["name"],
        "type": "revolute",
        "parent_body": j["parent"],
        "child_body": j["child"],
        "axis": [float(a) for a in j["axis"]],
        "location": {"value": [float(x) for x in j["origin"]], "unit": "m",
                     "provenance": dict(prov)},
        **({"limits": j["limits"]} if j["limits"] else {}),
    } for j in active_joints]

    return VehicleDefinition.model_validate({
        "name": name or root.get("name") or path.stem,
        "data_status": f"cad_import_spike (URDF: {path.name}); wings authored manually",
        "bodies": bodies,
        "joints": joint_specs,
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    })
