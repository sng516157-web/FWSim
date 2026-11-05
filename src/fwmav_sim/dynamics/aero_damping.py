"""Cycle-averaged aerodynamic body damping (M6 Task 8 fix; A-007 chain).

A constant cycle-averaged force removes the wings' dominant hover-stabilizing
mechanism: pitching or heaving the body changes the blade-element relative
airflow, which produces restoring forces (flapping counter-torque and heave
damping). This module computes the cycle-averaged damping DERIVATIVES from
the same time-resolved blade-element computation the mean forces come from
(brief §3.3 honesty: derived, never hand-formula'd):

    D = d(F_x, F_z, M_y)/d(v_z, omega_y)   about the B origin, central
    differences at the baseline motion

For the bio reference vehicle at the M1 hover condition the measured values
(order): dFz/dvz ~ -3.8e-2 N/(m/s), dMy/dwy ~ -9e-5 N m/(rad/s) — the pitch
damping is ~90x the bio vehicle's structural joint damping and is what makes
the Level-1 hover demo stabilizable at all (M6 report §3).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.kinematics.blade_element import WingFrameContext, discretize, element_state_ctx


def mean_wrench(vehicle, kinematics_map: dict, env, model, v_B, omega_B,
                steps_per_beat: int = 100) -> np.ndarray:
    """Cycle-mean generalized force [F_x, F_z, M_y] (B coords, about the B
    origin) with the body moving at constant (v_B, omega_B)."""
    R_wb = np.eye(3)
    elements = {w.name: discretize(w, vehicle.blade_elements.n_elements,
                                   vehicle.blade_elements.discretization)
                for w in vehicle.wings}
    f = next(iter(kinematics_map.values())).flap_frequency.value
    out = np.zeros(3)
    for w in vehicle.wings:
        kin = kinematics_map[w.name]
        for ti in np.arange(steps_per_beat) / (steps_per_beat * f):
            ctx = WingFrameContext(w, kin, ti, R_wb, np.zeros(3),
                                   R_wb @ omega_B, R_wb @ v_B, None)
            for el in elements[w.name]:
                st = element_state_ctx(w, el, ctx, env.nu)
                F = model.element_forces(st, env).resultant
                out += np.array([F[0], F[2], float(np.cross(st.r_W, F)[1])])
    return out / steps_per_beat


def body_motion_damping(vehicle, kinematics_map: dict, env,
                        eps: float = 0.05) -> dict[str, float]:
    """Central-difference damping derivatives about still air (see docstring).

    Returns keys dFx_dvz, dFz_dvz, dMy_dvz, dFx_dwy, dMy_dwy (heave/pitch
    damping matrix D: Q_damp = D @ [0, v_z, omega_y, 0]).
    """
    model = QuasiSteadyBladeElement(vehicle.aero)

    def wrench(v_B, w_B):
        return mean_wrench(vehicle, kinematics_map, env, model, v_B, w_B)

    dvz = (wrench(np.array([0.0, 0.0, eps]), np.zeros(3))
           - wrench(np.array([0.0, 0.0, -eps]), np.zeros(3))) / (2 * eps)
    dwy = (wrench(np.zeros(3), np.array([0.0, eps, 0.0]))
           - wrench(np.zeros(3), np.array([0.0, -eps, 0.0]))) / (2 * eps)
    return {
        "dFx_dvz": float(dvz[0]), "dFz_dvz": float(dvz[1]), "dMy_dvz": float(dvz[2]),
        "dFx_dwy": float(dwy[0]), "dFz_dwy": float(dwy[1]), "dMy_dwy": float(dwy[2]),
    }


def damping_matrix(d: dict[str, float]) -> np.ndarray:
    """(4,4) generalized damping D such that Q_damp = D @ qd for the reduced
    model's q = [x, z, theta, alpha_abd] (x/alpha channels zero in hover)."""
    D = np.zeros((4, 4))
    D[0, 1] = d["dFx_dvz"]
    D[0, 2] = d["dFx_dwy"]
    D[1, 1] = d["dFz_dvz"]
    D[1, 2] = d["dFz_dwy"]
    D[2, 1] = d["dMy_dvz"]
    D[2, 2] = d["dMy_dwy"]
    return D
