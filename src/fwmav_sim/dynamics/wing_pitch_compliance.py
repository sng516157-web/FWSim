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
        self.k = ps["stiffness"].value  # N m / rad
        self.c = ps["damping"].value  # N m s / rad
        self.rest = ps["rest_angle"].value  # rad
        if "inertia" in ps:
            self.I = ps["inertia"].value
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
