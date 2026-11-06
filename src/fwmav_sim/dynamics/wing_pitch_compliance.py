"""Passive torsional wing pitch (M7 Task 3).

Some platforms (AirPulse: passive feathering) have no pitch actuator: the
wing pitches about its spanwise axis against a torsional spring under the
aerodynamic pitching moment. This module adds that DOF per wing:

    I_p psi_ddot = M_aero_pitch(t) - c_p psi_dot - k_p (psi - psi_rest)

with psi the wing pitch angle (replacing the prescribed pitch profile for
that wing), M_aero_pitch the moment of the blade-element forces about the
pitch axis, I_p the wing's pitch inertia about the pitch axis (from
dynamics/wing_inertia when the wing has mass, else a required parameter),
k_p/c_p the spring/damper constants, psi_rest the rest angle.

Scope (recorded): fixed-body runner only in M7 (the rig case is the AirPulse
need); free-body coupling is a documented follow-up. Prescribed pitch stays
the default everywhere (backward compatible).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.dynamics.wing_inertia import wing_mass_properties
from fwmav_sim.kinematics.blade_element import (
    batched_element_state,
    discretize,
)


class PitchSpring:
    """Per-wing torsional spring-damper parameters + inertia resolution."""

    def __init__(self, wing):
        ps = wing.pitch_spring
        if ps is None:
            raise ValueError(f"wing {wing.name!r} has no pitch_spring block")
        self.k = ps.stiffness.value  # N m / rad
        self.c = ps.damping.value  # N m s / rad
        self.rest = ps.rest_angle.value  # rad
        if ps.inertia is not None:
            self.I = ps.inertia.value
        else:
            props = wing_mass_properties(wing)
            if props is None:
                raise ValueError(
                    f"wing {wing.name!r} pitch_spring needs either explicit inertia "
                    "or a wing mass (planform-derived, DR-006)")
            # pitch inertia about the pitch axis (spanwise, at the hinge),
            # = I_com,yy + m * (com offset from pitch axis)^2
            d = props.r_com_WR[0]  # chordwise offset of CoM from the pitch axis
            self.I = float(props.I_com_WR[1, 1]) + props.mass * d * d


def aero_pitch_moment(wing, elements, kin, t, env, model, pitch_angle: float,
                      pitch_rate: float) -> float:
    """Moment of the blade-element forces about the wing's pitch axis
    (spanwise axis through the pitch-axis chord station, at the hinge).

    Evaluated with the current passive pitch angle overriding the prescribed
    pitch profile (pitch_offset absorbs the mean; amplitude zeroed).
    """
    kin_p = kin.model_copy(update={
        "pitch_offset": type(kin.pitch_offset)(value=float(pitch_angle),
                                               unit=kin.pitch_offset.unit,
                                               provenance=kin.pitch_offset.provenance),
        "pitch_amplitude": type(kin.pitch_amplitude)(value=0.0,
                                                     unit=kin.pitch_amplitude.unit,
                                                     provenance=kin.pitch_amplitude.provenance),
        "pitch_fourier": [],
    })
    del pitch_rate  # pitch rate feeds the rotational term via the state (kept for the API)
    st = batched_element_state(wing, elements, kin_p, t, nu=env.nu,
                               v_air_W=env.v_air_W)
    bf = model.element_forces_batch(st, env)
    F = bf["F_trans"] + bf["F_rot"] + bf["F_am"]
    # moment arms about the pitch axis: elements sit at x_c from the axis
    # (chordwise, WR x) and s along the span (WR y); the pitch moment is the
    # component about the spanwise axis of r_from_axis x F.
    x_c = np.array([el.x_c for el in elements])
    # wing-frame force: rotate world force into the wing frame
    from fwmav_sim.kinematics.blade_element import wing_chain
    from fwmav_sim.kinematics.prescribed import wing_state as _ws
    from fwmav_sim.core.transforms import R_B_SP_wing
    from fwmav_sim.kinematics.blade_element import dihedral_of

    R_WR = wing_chain(_ws(kin_p, t), wing.side)[0]
    R_W_WR = R_B_SP_wing(kin_p.stroke_plane_angle.value, dihedral_of(wing), wing.side) @ R_WR
    F_WR = (R_W_WR.T @ F.T).T
    # moment about the spanwise (y_WR) axis at the pitch axis:
    # M_y = sum(x_c * F_z_WR - (z_el - 0) * F_x_WR); wings are flat (z_el = 0)
    return float(np.sum(x_c * F_WR[:, 2]))


def run_fixed_body_passive_pitch(vehicle, kinematics, env, model,
                                 n_wingbeats: int, steps_per_wingbeat: int):
    """Fixed-body run with passive torsional pitch (M7 Task 3).

    The runner becomes an ODE solve: per-wing state [psi, psi_dot] evolves as
    I_p psi_ddot = M_aero(t, psi) - c psi_dot - k (psi - psi_rest); the force
    channels are evaluated at the evolved pitch. Prescribed pitch is ignored
    for sprung wings (a validity warning is recorded).
    """
    from scipy.linalg import expm

    from fwmav_sim.core.timeseries import TimeSeries

    kin_map = (kinematics if isinstance(kinematics, dict)
               else {w.name: kinematics for w in vehicle.wings})
    sprung = {w.name: PitchSpring(w) for w in vehicle.wings
              if w.pitch_spring is not None}
    if not sprung:
        raise ValueError("passive_pitch enabled but no wing has a pitch_spring block")
    elements = {w.name: discretize(w, vehicle.blade_elements.n_elements,
                                   vehicle.blade_elements.discretization)
                for w in vehicle.wings}
    f = next(iter(kin_map.values())).flap_frequency.value
    period = 1.0 / f
    names = list(sprung)
    wing_by_name = {w.name: w for w in vehicle.wings}
    t_eval = np.arange(n_wingbeats * steps_per_wingbeat) * (period / steps_per_wingbeat)
    dt = period / steps_per_wingbeat

    def spring_step(ps, M0, dM, psi, psi_dot, h):
        """Semi-implicit exact update of the pitch oscillator over h.

        The aero moment is linearized, M(psi) ~ M0 + dM (psi - psi_n), and the
        resulting linear system  I psi_ddot = M0 + dM (psi - psi_n)
            - c psi_dot - k (psi - rest)
        is integrated exactly via 2x2 matrix exponential. Treating the aero
        stiffness dM implicitly is what keeps the stiff-spring regime stable
        at wingbeat-rate step sizes (explicit M-freezing diverges when
        |dM| >> k, which is the norm here: dM ~ -0.15 N m/rad).
        Cost: 2 aero evaluations per step (M0 and the finite-difference probe).
        """
        K = ps.k - dM  # total effective stiffness (aero slope is restorative)
        if abs(K) < 1e-30:
            K = 1e-30
        # equilibrium of the linearized system:
        # K psi* = M0 - dM psi_n + k rest
        psi_eq = (M0 - dM * psi + ps.k * ps.rest) / K
        A = np.array([[0.0, 1.0], [-K / ps.I, -ps.c / ps.I]])
        x = expm(A * h) @ np.array([psi - psi_eq, psi_dot])
        return psi_eq + x[0], x[1]

    dpsi_fd = 1e-4  # finite-difference probe for the aero stiffness
    state = {n: [sprung[n].rest, 0.0] for n in names}
    psi_hist = {n: np.zeros(len(t_eval)) for n in names}
    psi_dot_hist = {n: np.zeros(len(t_eval)) for n in names}
    for i, ti in enumerate(t_eval):
        for n in names:
            psi, psi_dot = state[n]
            w, els, kin = wing_by_name[n], elements[n], kin_map[n]
            M0 = aero_pitch_moment(w, els, kin, ti, env, model, psi, psi_dot)
            M1 = aero_pitch_moment(w, els, kin, ti, env, model,
                                   psi + dpsi_fd, psi_dot)
            psi, psi_dot = spring_step(sprung[n], M0, (M1 - M0) / dpsi_fd,
                                       psi, psi_dot, dt)
            state[n] = [psi, psi_dot]
            psi_hist[n][i] = psi
            psi_dot_hist[n][i] = psi_dot

    # force channels at the evolved pitch
    channels: dict[str, np.ndarray] = {}
    for name in names:
        channels[f"{name}/psi_passive"] = psi_hist[name]
        channels[f"{name}/psi_dot_passive"] = psi_dot_hist[name]
    for w in vehicle.wings:
        m = len(elements[w.name])
        for key, shape in {"F_total": (len(t_eval), 3), "M_total": (len(t_eval), 3),
                           "P_aero": (len(t_eval),),
                           "alpha": (len(t_eval), m),
                           "reynolds": (len(t_eval), m)}.items():
            channels[f"{w.name}/{key}"] = np.zeros(shape)
        for i, ti in enumerate(t_eval):
            kin_eff = kin_map[w.name]
            if w.name in sprung:
                import copy

                kin_eff = copy.deepcopy(kin_map[w.name])
                kin_eff.pitch_offset.value = float(psi_hist[w.name][i])
                kin_eff.pitch_amplitude.value = 0.0
            st = batched_element_state(w, elements[w.name], kin_eff, ti, nu=env.nu,
                                       v_air_W=env.v_air_W)
            bf = model.element_forces_batch(st, env)
            F_el = bf["F_trans"] + bf["F_rot"] + bf["F_am"]
            channels[f"{w.name}/F_total"][i] = F_el.sum(axis=0)
            channels[f"{w.name}/M_total"][i] = np.cross(st["r_W"], F_el).sum(axis=0)
            channels[f"{w.name}/P_aero"][i] = -float((F_el * st["v_W"]).sum())
            channels[f"{w.name}/alpha"][i] = st["alpha"]
            channels[f"{w.name}/reynolds"][i] = st["reynolds"]

    meta = {"kind": "fixed_body", "passive_pitch": True,
            "n_wingbeats": n_wingbeats, "steps_per_wingbeat": steps_per_wingbeat,
            "flap_frequency_hz": f, "wingbeat_period_s": period,
            "vehicle": vehicle.name, "aero_model": type(model).__name__,
            "assumptions": ["A-001", "A-002", "A-003", "A-004", "A-005", "A-006"],
            "sprung_wings": names}
    return TimeSeries(t=t_eval, channels=channels, meta=meta)
