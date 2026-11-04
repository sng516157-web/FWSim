"""AirPulse rig comparison (M4): our fixed-body run vs the published
phase-averaged force coefficients of Gu et al. 2026 (arXiv:2602.06811).

Their normalization (Eqs. 15-18 of the paper, transcribed from the text):
  r2    = sqrt( (1/S) integral r^2 dS )     (radius of 2nd moment of area)
  U_ref = 2 * Phi * f * r2                  (Phi = PEAK-TO-PEAK amplitude, rad)
  C_Fk  = F_k / (0.5 * rho * U_ref^2 * S)

Their rig: phase-averaged over 40 cycles, 222 Hz sampling, 4th-order
Butterworth 50 Hz lowpass, inertial baseline subtraction (their aero
estimate excludes wing inertia — our wings are massless, so our force is
aero-only too, but by construction; the approaches differ and that is
recorded as a limitation, not hidden).

Frame note: paper uses FLU (+z up); we use NED (z down): C_Fz sign flips.

This is a coarse literature comparison (published point values), labelled as
such — not a validation dataset (no time series, no uncertainty bands beyond
the paper's phase-std error bars).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.datasets.filtering import FilterSpec, apply_filter
from fwmav_sim.experiment.record import load_record
from fwmav_sim.kinematics.blade_element import discretize

#: paper's published cycle-averaged coefficients, four-vein full wing
#: (C_Fx, C_Fz) — arXiv:2602.06811 Fig. 2H / text
PAPER_FULL_WING = {"C_Fx": 6.084, "C_Fz": 4.780}
PAPER_R2_FULL = 0.128  # m (their full-wing radius of 2nd moment of area)


def wing_second_moment_radius(wing, n_elements: int = 64) -> float:
    """r2 of our wing geometry from the actual discretization (Eq. 15)."""
    elements = discretize(wing, n_elements)
    hinge_y = abs(wing.hinge_location.value[1])
    num = sum((hinge_y + el.s) ** 2 * el.area for el in elements)
    den = sum(el.area for el in elements)
    return float(np.sqrt(num / den))


def rig_comparison(record_dir: str | Path, out: str | Path | None = None) -> dict:
    """Compute our coefficients from a run record and diff vs the paper."""
    ts, meta = load_record(record_dir)
    from fwmav_sim.experiment.config import load_experiment, load_vehicle

    exp = load_experiment(Path(record_dir) / "config.yaml")
    vehicle = load_vehicle(Path(record_dir) / "vehicle.yaml")
    kin = exp.kinematics
    assert not isinstance(kin, dict)
    f = kin.flap_frequency.value
    phi_pp = 2.0 * kin.flap_amplitude.value  # their Phi is peak-to-peak
    rho = exp.environment.air_density.value

    T = ts.meta["wingbeat_period_s"]
    filt = FilterSpec(order=4, cutoff_hz=50.0,
                      rationale="matching the paper's rig processing "
                      "(4th-order Butterworth, 50 Hz)")

    per_wing = {}
    for wing in vehicle.wings:
        r2 = wing_second_moment_radius(wing)
        S = sum(el.area for el in discretize(wing, vehicle.blade_elements.n_elements))
        U_ref = 2.0 * phi_pp * f * r2
        norm = 0.5 * rho * U_ref**2 * S
        Fx = apply_filter(ts.t, ts[f"{wing.name}/F_total"][:, 0], filt)
        Fz = apply_filter(ts.t, ts[f"{wing.name}/F_total"][:, 2], filt)
        Fx_mean, _ = wingbeat_mean(ts.__class__(t=ts.t, channels={"F": Fx}), "F", T)
        Fz_mean, _ = wingbeat_mean(ts.__class__(t=ts.t, channels={"F": Fz}), "F", T)
        per_wing[wing.name] = {
            "C_Fx": float(Fx_mean / norm),
            "C_Fz": float(-Fz_mean / norm),  # FLU z-up conversion
            "r2_m": r2, "S_m2": S, "U_ref_mps": U_ref,
        }

    ours = {k: float(np.mean([w[k] for w in per_wing.values()])) for k in ("C_Fx", "C_Fz")}
    result = {
        "paper": "Gu et al. 2026 (arXiv:2602.06811), Fig. 2H, four-vein full wing",
        "paper_values": PAPER_FULL_WING,
        "ours": ours,
        "per_wing": per_wing,
        "residuals": {k: ours[k] - PAPER_FULL_WING[k] for k in ours},
        "relative_residuals": {k: (ours[k] - PAPER_FULL_WING[k]) / PAPER_FULL_WING[k]
                               for k in ours},
        "filter": filt.model_dump(),
        "caveats": [
            "coarse literature comparison against published point values; no time "
            "series, no uncertainty bands: NOT a validation dataset",
            "their aero estimate uses inertial baseline subtraction; ours is "
            "aero-only by construction (massless wings)",
            "our pitch kinematics are a placeholder for their passive feathering",
            "our quasi-steady coefficients are Robofly-regime; AirPulse operates "
            "at Re ~ 3e4 (extrapolation flagged in the vehicle file)",
        ],
    }
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys

    res = rig_comparison(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps({k: res[k] for k in ("ours", "paper_values", "relative_residuals")},
                     indent=2))
