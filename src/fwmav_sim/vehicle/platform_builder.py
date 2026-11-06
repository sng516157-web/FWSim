"""Platform builder (M8 Task 3): assemble a vehicle from a kinematic tree.

The platform is a TREE (M8 design rule 3 — mechanical topology is a tree,
not a free graph): bodies -> joints -> wings. This module is the Qt-free
core the tree editor drives; it is also usable headless:

    from fwmav_sim.vehicle.platform_builder import PlatformBuilder
    pb = PlatformBuilder.from_urdf("mycobot.urdf")
    pb.add_wing("wing_right", side="right", span=0.3, chord=0.18, ...)
    pb.save_vehicle("reference_vehicles/mycobot.yaml")

Everything compiles to the SAME verbose vehicle YAML (M8 design rule 1):
the builder emits full provenance blocks, never the compact form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated",
         "reference": "platform builder (M8); edited by hand or from CAD"}


def _p(value, unit, prov=None):
    return {"value": value, "unit": unit, "provenance": prov or dict(_PROV)}


class PlatformBuilder:
    """Editable vehicle tree: bodies + joints (+ wings) -> VehicleDefinition."""

    def __init__(self, name: str):
        self.name = name
        self.bodies: list[dict[str, Any]] = []
        self.joints: list[dict[str, Any]] = []
        self.wings: list[dict[str, Any]] = []
        self.blade_elements = {"n_elements": 16, "discretization": "equal_span"}
        self.aero: dict[str, Any] | None = None

    # --- constructors --------------------------------------------------------

    @classmethod
    def from_urdf(cls, path: str | Path, name: str | None = None) -> "PlatformBuilder":
        """URDF import as a node source (M6.75 importer): bodies + joints
        come from the CAD export; wings are attached by hand afterwards."""
        from fwmav_sim.vehicle.urdf_import import import_urdf

        vd = import_urdf(path, name=name)
        pb = cls(vd.name)
        pb.bodies = [b.model_dump() for b in vd.bodies]
        pb.joints = [j.model_dump() for j in vd.joints]
        return pb

    @classmethod
    def from_vehicle(cls, path: str | Path) -> "PlatformBuilder":
        from fwmav_sim.experiment.config import load_vehicle

        vd = load_vehicle(path)
        pb = cls(vd.name)
        pb.bodies = [b.model_dump() for b in vd.bodies]
        pb.joints = [j.model_dump() for j in vd.joints]
        pb.wings = [w.model_dump() for w in vd.wings]
        pb.blade_elements = vd.blade_elements.model_dump()
        pb.aero = vd.aero.model_dump()
        return pb

    # --- tree editing ----------------------------------------------------------

    def add_body(self, name: str, mass: float, com_offset=(0.0, 0.0, 0.0),
                 inertia=None, provenance: dict | None = None) -> None:
        if any(b["name"] == name for b in self.bodies):
            raise ValueError(f"duplicate body {name!r}")
        prov = provenance or dict(_PROV)
        self.bodies.append({
            "name": name,
            "mass": _p(mass, "kg", prov),
            "com_offset": _p(list(com_offset), "m", prov),
            "inertia": _p(inertia or [[1e-5, 0, 0], [0, 1e-5, 0], [0, 0, 1e-5]],
                          "kg m^2", prov),
        })

    def add_joint(self, name: str, joint_type: Literal["revolute", "prismatic"],
                  parent: str, child: str, axis, origin=(0.0, 0.0, 0.0),
                  limits=None, provenance: dict | None = None) -> None:
        names = {b["name"] for b in self.bodies}
        if parent not in names or child not in names:
            raise ValueError(f"joint {name!r}: parent/child must be existing bodies "
                             f"(have: {sorted(names)})")
        if any(j["name"] == name for j in self.joints):
            raise ValueError(f"duplicate joint {name!r}")
        prov = provenance or dict(_PROV)
        spec: dict[str, Any] = {
            "name": name, "type": joint_type, "parent_body": parent,
            "child_body": child,
            "axis": [float(a) for a in axis],
            "location": _p(list(origin), "m", prov),
        }
        if limits is not None:
            spec["limits"] = [float(limits[0]), float(limits[1])]
        self.joints.append(spec)

    def add_wing(self, name: str, side: Literal["left", "right"], span: float,
                 root_chord: float, tip_chord: float | None = None,
                 hinge_location=(0.0, 0.0, 0.0), dihedral_angle: float = 0.0,
                 pitch_axis_fraction: float = 0.25,
                 planform: str = "rectangular",
                 provenance: dict | None = None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if any(w["name"] == name for w in self.wings):
            raise ValueError(f"duplicate wing {name!r}")
        prov = provenance or dict(_PROV)
        self.wings.append({
            "name": name, "side": side,
            "span": _p(span, "m", prov),
            "root_chord": _p(root_chord, "m", prov),
            "tip_chord": _p(tip_chord if tip_chord is not None else root_chord,
                            "m", prov),
            "planform": planform,
            "hinge_location": _p(list(hinge_location), "m", prov),
            "pitch_axis_fraction": _p(pitch_axis_fraction, "1", prov),
            "dihedral_angle": _p(dihedral_angle, "rad", prov),
        })

    def remove_wing(self, name: str) -> None:
        if not any(w["name"] == name for w in self.wings):
            raise ValueError(f"no wing {name!r}")
        self.wings = [w for w in self.wings if w["name"] != name]

    # --- compile ---------------------------------------------------------------

    def to_vehicle_dict(self) -> dict[str, Any]:
        if not self.bodies:
            raise ValueError("platform has no bodies")
        aero = self.aero or {
            "name": "quasi_steady_blade_element",
            "coefficients": {
                k: _p(v, u, {"source": "literature",
                             "reference": "Dickinson et al. 1999 (Robofly fits); "
                                          "validity for this vehicle unverified"})
                for k, v, u in [
                    ("cl_0", 0.225, "1"), ("cl_slope", 1.58, "1"),
                    ("cl_freq", 2.13, "1"), ("cl_phase_deg", 7.20, "deg"),
                    ("cd_0", 1.92, "1"), ("cd_slope", 1.55, "1"),
                    ("cd_freq", 2.04, "1"), ("cd_phase_deg", 9.82, "deg")]
            },
        }
        return {
            "name": self.name,
            "data_status": "estimated",
            "bodies": self.bodies,
            "joints": self.joints,
            "wings": self.wings,
            "blade_elements": dict(self.blade_elements),
            "aero": aero,
        }

    def to_vehicle(self) -> VehicleDefinition:
        """Validate through the schema — compile errors surface HERE, not at
        run time."""
        return VehicleDefinition.model_validate(self.to_vehicle_dict())

    def save_vehicle(self, path: str | Path) -> Path:
        vd = self.to_vehicle()  # validate first
        path = Path(path)
        path.write_text(
            f"# Vehicle authored with the platform builder (M8 Task 3).\n"
            f"# Validated against the schema at export (name: {vd.name}).\n"
            + yaml.safe_dump(self.to_vehicle_dict(), sort_keys=False),
            encoding="utf-8")
        return path
