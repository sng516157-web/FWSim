"""Compact vehicle authoring (M6.5 Task 3).

Problem: the full provenance-wrapped YAML is the correct *storage* format
(every value carries its source, brief §5.8) but an unreadable *authoring*
format — twenty physical values become two hundred lines. The compact
dialect restores authoring ergonomics:

    format: compact-v1
    name: my_mav
    provenance: {source: estimated, uncertainty_or_range: "draft, unmeasured"}
    bodies:
      - name: thorax
        mass: 0.011            # kg (units from the field registry)
        com_offset: [0, 0, 0]  # m
        inertia: [[2e-6,0,0],[0,2e-6,0],[0,0,2e-6]]  # kg m^2
    ...

``load_vehicle`` detects ``format: compact-v1`` and expands it to the full
schema before validation: the block-level ``provenance`` (default
``source: estimated``) is stamped onto every parameter; per-field overrides
are honoured (any field written in the full nested form passes through).
Units come from the field registry below (SI, matching the schema comments).

Rules (rubric): exists because authoring by hand is the main entry path;
integrates at the load boundary only (nothing downstream changes); the
verbose form remains the on-disk truth after first save.
"""

from __future__ import annotations

from typing import Any

#: field path -> (unit, kind) for every physical field in the compact dialect
_UNITS: dict[str, tuple[str, str]] = {
    "mass": ("kg", "scalar"),
    "com_offset": ("m", "vec3"),
    "inertia": ("kg m^2", "mat3"),
    "location": ("m", "vec3"),
    "damping": ("N m s / rad", "scalar"),
    "span": ("m", "scalar"),
    "root_chord": ("m", "scalar"),
    "tip_chord": ("m", "scalar"),
    "hinge_location": ("m", "vec3"),
    "pitch_axis_fraction": ("1", "scalar"),
    "dihedral_angle": ("rad", "scalar"),
    "area": ("m^2", "scalar"),
    "cp_offset": ("m", "scalar"),
    "air_density": ("kg/m^3", "scalar"),
    "kinematic_viscosity": ("m^2/s", "scalar"),
    "flap_amplitude": ("rad", "scalar"),
    "flap_frequency": ("Hz", "scalar"),
    "pitch_amplitude": ("rad", "scalar"),
    "pitch_phase": ("rad", "scalar"),
    "pitch_offset": ("rad", "scalar"),
    "stroke_plane_angle": ("rad", "scalar"),
    "stroke_phase": ("rad", "scalar"),
    "deviation_amplitude": ("rad", "scalar"),
    "deviation_phase": ("rad", "scalar"),
    "cl_0": ("1", "scalar"), "cl_slope": ("1", "scalar"), "cl_freq": ("1", "scalar"),
    "cl_phase_deg": ("deg", "scalar"), "cd_0": ("1", "scalar"), "cd_slope": ("1", "scalar"),
    "cd_freq": ("1", "scalar"), "cd_phase_deg": ("deg", "scalar"),
    "cn": ("1", "scalar"),
}

COMPACT_FORMAT = "compact-v1"


def is_compact(raw: dict) -> bool:
    return raw.get("format") == COMPACT_FORMAT


def _wrap(value: Any, field: str, prov: dict) -> dict:
    """Wrap a compact scalar/vector/matrix into the full Parameter form."""
    unit, _kind = _UNITS[field]
    if isinstance(value, dict) and "value" in value:
        return value  # already the full form — pass through
    return {"value": value, "unit": unit, "provenance": dict(prov)}


def _expand_block(block: dict, fields: tuple[str, ...], prov: dict) -> dict:
    out = {}
    for key, val in block.items():
        if key in fields:
            out[key] = _wrap(val, key, prov)
        elif isinstance(val, dict) and "value" not in val and key == "provenance":
            out[key] = val
        else:
            out[key] = val
    return out


def expand_compact(raw: dict) -> dict:
    """Expand a compact-v1 vehicle mapping into the full schema mapping."""
    prov = {"source": "estimated", **(raw.get("provenance") or {})}
    out: dict[str, Any] = {k: v for k, v in raw.items()
                           if k not in ("format", "provenance")}

    body_fields = ("mass", "com_offset", "inertia")
    joint_fields = ("location", "damping")
    wing_fields = ("span", "root_chord", "tip_chord", "hinge_location",
                   "pitch_axis_fraction", "dihedral_angle", "mass")
    out["bodies"] = [_expand_block(b, body_fields, prov) for b in raw.get("bodies", [])]
    out["joints"] = [_expand_block(j, joint_fields, prov) for j in raw.get("joints", [])]
    out["wings"] = [_expand_block(w, wing_fields, prov) for w in raw.get("wings", [])]

    if "aero" in raw:
        aero = dict(raw["aero"])
        if "coefficients" in aero:
            aero["coefficients"] = {
                k: _wrap(v, k, prov) for k, v in aero["coefficients"].items()
            }
        out["aero"] = aero
    if "abdomen_aero" in raw:
        out["abdomen_aero"] = _expand_block(raw["abdomen_aero"], ("area", "cp_offset"), prov)
        if "coefficients" in raw["abdomen_aero"]:
            out["abdomen_aero"]["coefficients"] = {
                k: _wrap(v, k, prov) for k, v in raw["abdomen_aero"]["coefficients"].items()
            }
    return out


def expand_experiment(raw: dict) -> dict:
    """Expand compact kinematics/environment blocks in an experiment mapping."""
    out = {k: v for k, v in raw.items() if k not in ("format", "provenance")}
    prov = {"source": "estimated", **(raw.get("provenance") or {})}
    kin_fields = ("flap_amplitude", "flap_frequency", "pitch_amplitude",
                  "pitch_phase", "pitch_offset", "stroke_plane_angle",
                  "stroke_phase", "deviation_amplitude", "deviation_phase")
    env_fields = ("air_density", "kinematic_viscosity")
    if "kinematics" in raw and isinstance(raw["kinematics"], dict):
        kin = raw["kinematics"]
        if "flap_amplitude" in kin:  # broadcast form
            out["kinematics"] = _expand_block(kin, kin_fields, prov)
        else:  # per-wing mapping
            out["kinematics"] = {w: _expand_block(k, kin_fields, prov)
                                 for w, k in kin.items()}
    if "environment" in raw:
        out["environment"] = _expand_block(raw["environment"], env_fields, prov)
    return out
