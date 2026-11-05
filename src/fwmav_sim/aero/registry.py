"""Named aerodynamic coefficient sets (M6 Task 6).

Problem: the wing aero coefficients were a single inline block per vehicle —
always Dickinson 1999 Robofly fits, silently extrapolated to whatever regime
the vehicle actually flies in (the AirPulse Re mismatch is the M4 lesson).

Design: coefficient sets are first-class named data with regime metadata and
citations. A vehicle YAML selects a set by name (``aero.coefficient_set``)
or keeps inline ``coefficients`` (fully backward compatible — exactly one of
the two must be present). The run record's validity section names the set
and its regime so the choice is visible in every output.

Sets shipped:
- ``dickinson1999_robofly``: the Dickinson et al. 1999 (Science 284:1954)
  Robofly fits, Re ~ O(100). The M1–M5 default. Form: ``dickinson1999``.
- ``flatplate_poststall_classical``: the classical post-stall flat-plate
  model C_L = c_n sin^2(a) cos(a), C_D = c_n sin^3(a), c_n = 2.0 (inviscid +
  separated-flow flat plate, the widely used high-alpha model for Re ~ 1e3+;
  see Tani & Ward 1969 and the Jones & Babinsky 2010 review). Evaluated
  EXACTLY (form ``poststall_classical``) — not shoehorned into the Dickinson
  single-sine fit, which fits this shape only to ~0.12.
- ``dickson2006_hummingbird``: the Dickinson-lab integrated-model fits
  C_L = 1.8 sin(2a), C_D = 1.95 - 1.5 cos(2a) (Dickson, Straw, Poelma &
  Dickinson 2006, AIAA-2006-34), the set the Purdue flappy simulator
  (Fei et al. 2019, ICRA) implements and validated on its hummingbird rig.
  NOTE on d_cp: flappy's chordwise centre-of-pressure fit exists because
  its model is wing-integrated; FWSim's per-element force distribution
  produces moments directly, so no CP model is needed here (recorded, not
  implemented).
"""

from __future__ import annotations

from fwmav_sim.vehicle.schema import Parameter, Provenance

#: name -> set descriptor; form selects the evaluation path in quasi_steady.py
_SETS: dict[str, dict] = {
    "dickinson1999_robofly": {
        "form": "dickinson1999",
        "coefficients": {
            "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
            "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
        },
        "regime": "Robofly, Re ~ O(100)",
        "reference": "Dickinson et al. 1999, Science 284:1954",
        "source": "literature",
    },
    "flatplate_poststall_classical": {
        "form": "poststall_classical",
        "coefficients": {"cn": 2.0},
        "regime": "flat plate, post-stall / high-alpha, Re ~ 1e3+ (inviscid+separated)",
        "reference": ("classical post-stall flat-plate model (Tani & Ward 1969; "
                      "Jones & Babinsky 2010 review); cn = 2.0 nominal"),
        "source": "literature",
    },
    "dickson2006_hummingbird": {
        "form": "dickinson1999",
        "coefficients": {
            # C_L = 1.8 sin(2a)  =>  0 + 1.8 sin(2 a_deg - 0)
            # C_D = 1.95 - 1.5 cos(2a)  =>  1.95 - 1.5 cos(2 a_deg - 0)
            "cl_0": 0.0, "cl_slope": 1.8, "cl_freq": 2.0, "cl_phase_deg": 0.0,
            "cd_0": 1.95, "cd_slope": 1.5, "cd_freq": 2.0, "cd_phase_deg": 0.0,
        },
        "regime": "hummingbird-scale flapping (validated on the Purdue rig, Fei 2019)",
        "reference": ("Dickson, Straw, Poelma & Dickinson 2006, 44th AIAA "
                      "Aerospace Sciences Meeting, AIAA-2006-34 (as implemented "
                      "and rig-validated by Fei et al. 2019, flappy ICRA)"),
        "source": "literature",
    },
}


def available_sets() -> list[str]:
    return sorted(_SETS)


def coefficient_set_info(name: str) -> dict:
    if name not in _SETS:
        raise ValueError(f"unknown coefficient set {name!r}; available: {available_sets()}")
    return _SETS[name]


def resolve_coefficients(spec) -> tuple[dict[str, Parameter], dict | None]:
    """(coefficients, set_info) from an AeroModelSpec.

    Exactly one of inline coefficients / coefficient_set must be present.
    set_info is None for inline coefficients (Dickinson form implied).
    """
    inline = bool(spec.coefficients)
    named = getattr(spec, "coefficient_set", None) is not None
    if inline == named:
        raise ValueError(
            "aero: exactly one of 'coefficients' (inline) or 'coefficient_set' "
            "(named registry set) must be given")
    if inline:
        return spec.coefficients, None
    info = coefficient_set_info(spec.coefficient_set)
    prov = Provenance(source=info["source"], reference=info["reference"],
                      valid_regime=info["regime"])
    return {k: Parameter(value=float(v), unit="1" if not k.endswith("_deg") else "deg",
                         provenance=prov)
            for k, v in info["coefficients"].items()}, info
