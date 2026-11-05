"""M7 Task 2: three-way comparison on the AirPulse rig case.

Our quasi-steady blade element vs the UVLM backend (PteraSoftware, fixed
wake) vs the paper's published coefficients (Gu et al. 2026). The question
(M4 follow-up): does the C_F gap move when wake/induced physics is included?

Caveats (recorded, M7 report): UVLM tier uses a thin airfoil (naca0008) and
prescribed pitch (AirPulse pitch is passive); prescribed wake is the stable
tier (free wake rolls up unstably at these amplitudes).

Run: python derivations/m7_uvlm_vs_quasisteady.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.aero.uvlm_backend import run_uvlm_fixed_body
    from fwmav_sim.analysis.averaging import wingbeat_mean
    from fwmav_sim.experiment.config import kinematics_map, load_experiment, load_vehicle
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    # NOTE: AirPulse rig conditions (70 deg amplitude + 50 deg dihedral)
    # exceed ptera's movement range (+-90 deg wing-axes): the clap regime is
    # outside the UVLM wrapper's validity (recorded in the M7 report). The
    # comparison case is the bio hover (60 deg, no dihedral).
    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    kin_map = kinematics_map(exp, vehicle)
    period = 1.0 / exp.kinematics.flap_frequency.value

    # ours (quasi-steady blade element)
    ts = run_fixed_body(vehicle, kin_map, env, QuasiSteadyBladeElement(vehicle.aero),
                        exp.n_wingbeats, exp.steps_per_wingbeat)
    qs = {}
    for w in vehicle.wings:
        mean, _ = wingbeat_mean(ts, f"{w.name}/F_total", period)
        qs[w.name] = np.asarray(mean)
    qs_total = sum(qs.values())

    # UVLM
    out = run_uvlm_fixed_body(vehicle, kin_map, env,
                              exp.n_wingbeats, exp.steps_per_wingbeat // 4)
    t, F = out["t"], out["forces_W"]
    last = t >= t[-1] - period
    uvlm_mean = F[last].mean(axis=0)

    print("=== three-way comparison, AirPulse rig conditions ===")
    print(f"quasi-steady total F (NED): [{qs_total[0]:+.4f}, {qs_total[1]:+.4f}, "
          f"{qs_total[2]:+.4f}] N")
    print(f"UVLM total F (wind~NED):    [{uvlm_mean[0]:+.4f}, {uvlm_mean[1]:+.4f}, "
          f"{uvlm_mean[2]:+.4f}] N")
    print(f"paper coefficients imply: C_Fx=6.084 C_Fz=4.780 (see M4 report)")
    ratio = qs_total[2] / uvlm_mean[2] if abs(uvlm_mean[2]) > 1e-9 else float("nan")
    print(f"ratio qs/uvlm (Fz): {ratio:.3f}")


if __name__ == "__main__":
    main()
