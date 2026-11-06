"""Wing structural mass in the free-body and fixed-body models (post-M5 item 1;
decision DR-006, assumption A-010).

Wings follow prescribed kinematics (A-004), so wing mass adds no dynamic
DoFs: each wing is a kinematically driven rigid appendage of the thorax. Its
inertial effect enters exactly by folding the wing's spatial inertia into the
base block of the mass matrix and adding the velocity/prescribed/gravity part
of the wing reaction wrench to the right-hand side:

    (M_bodies + sum_w M_w(q, t)) u_dot = tau + Q_ext - bias + sum_w Q_w(q, u, t)

with, per wing (all derived from the frozen transform chain of
kinematics/blade_element.py — no finite differences):

    M_w   = spatial inertia of the wing about the base origin (base coords)
    Q_w   = [n; f] reaction wrench on the thorax from
            F = -m (a_c^0 + R_wb (omega_B x v_B) - g_W)   (force at wing CoM)
            N = -(I_cW alpha_presc + omega_w x I_cW omega_w)  (moment about CoM)

where a_c^0 is the wing CoM acceleration with base acceleration zero (the
element-kinematics acceleration of blade_element.py), alpha_presc the wing
angular acceleration with base angular acceleration zero, and g_W =
[0, 0, +9.81] (z down, frames.md §1.1). This is the consistent treatment of
the algebraic loop noted in A-009: the u_dot-dependent part of the wing
reaction moves to the left-hand side (M_w), so no implicit solve is needed.

Wing mass properties derive from the planform as a thin flat plate of uniform
areal density (A-010), integrated analytically per planform; the pitch-axis
offset of the chordwise CoM is included via pitch_axis_fraction.

Scope: Level-2 free body (simulation/free_body.py) and fixed-body rig
records (simulation/fixed_body.py, where the inertial root force is what a
physical force balance would measure on top of the aerodynamic force). The
Level-1 reduced longitudinal model is unchanged: it consumes cycle-averaged
forces only (A-007), and the inertial force of a symmetric wingbeat is
zero-mean over a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwmav_sim.core.transforms import R_B_SP_wing, cross3, skew
from fwmav_sim.kinematics.blade_element import dihedral_of, wing_chain_with_alpha
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics, wing_state
from fwmav_sim.vehicle.schema import WingSpec

G_W = np.array([0.0, 0.0, 9.81])  # world gravity vector, z down (frames.md §1.1)


# --- analytic planform integrals (A-010) --------------------------------------
# c(s) = local chord at span station s in [0, R]; all integrals over the span.

def _planform_integrals(wing: WingSpec) -> tuple[float, float, float, float, float, float]:
    """Integrals over the span of c, s*c, s^2*c, c^2, c^3, s*c^2 (c = chord)."""
    R = wing.span.value
    c_r, c_t = wing.root_chord.value, wing.tip_chord.value
    if wing.planform == "rectangular":
        return (c_r * R, c_r * R**2 / 2.0, c_r * R**3 / 3.0,
                c_r**2 * R, c_r**3 * R, c_r**2 * R**2 / 2.0)
    if wing.planform == "trapezoidal":
        dc = c_t - c_r
        i_c1 = R * (c_r + c_t) / 2.0
        i_sc = R**2 * (c_r / 2.0 + dc / 3.0)
        i_s2c = R**3 * (c_r / 3.0 + dc / 4.0)
        i_c2 = R * (c_r**2 + c_r * c_t + c_t**2) / 3.0
        i_c3 = R * (c_t**4 - c_r**4) / (4.0 * dc) if abs(dc) > 1e-15 else c_r**3 * R
        i_sc2 = R**2 * (c_r**2 / 2.0 + 2.0 * c_r * dc / 3.0 + dc**2 / 4.0)
        return i_c1, i_sc, i_s2c, i_c2, i_c3, i_sc2
    if wing.planform == "elliptical":
        return (c_r * R * np.pi / 4.0, c_r * R**2 / 3.0, c_r * R**3 * np.pi / 16.0,
                c_r**2 * R * 2.0 / 3.0, c_r**3 * R * 3.0 * np.pi / 16.0,
                c_r**2 * R**2 / 4.0)
    raise ValueError(f"unknown planform {wing.planform!r}")


def _custom_planform_integrals(wing: WingSpec):
    """Mass-property integrals for a custom (station-defined, possibly swept)
    outline, by Gauss quadrature over the station panels.

    Returns (i_c1, i_sc, i_s2c, i_c3, i_xc, i_x2c, i_sxc) — integrals over the
    span of c, s·c, s²·c, c³, x·c, x²·c, s·x·c, where x(s) = le_offset(s) −
    c(s)/2 is the mid-chord (element-centre) line. The plate's own chordwise
    inertia adds c²/12 per unit length (the i_c3 term)."""
    from fwmav_sim.kinematics.blade_element import chord_at, le_offset_at

    st = wing.planform_stations
    # 8-point Gauss-Legendre per panel: exact for polynomials up to degree 15
    # (c is linear per panel; the integrands are at most degree 5)
    gx, gw = np.polynomial.legendre.leggauss(8)
    i_c1 = i_sc = i_s2c = i_c3 = i_xc = i_x2c = i_sxc = 0.0
    for a, b in zip(st, st[1:]):
        half, mid = 0.5 * (b.y - a.y), 0.5 * (a.y + b.y)
        s_pts = mid + half * gx
        for s, w in zip(s_pts, gw * half, strict=True):
            c = chord_at(wing, s)
            x = le_offset_at(wing, s) - 0.5 * c
            i_c1 += w * c
            i_sc += w * s * c
            i_s2c += w * s * s * c
            i_c3 += w * c**3
            i_xc += w * x * c
            i_x2c += w * x * x * c
            i_sxc += w * s * x * c
    return i_c1, i_sc, i_s2c, i_c3, i_xc, i_x2c, i_sxc


@dataclass(frozen=True)
class WingMassProperties:
    """Thin-flat-plate mass properties of one wing (uniform areal density).

    Axes follow the wing-root frame WR of blade_element.py: x toward the
    leading edge (chordwise), y spanwise (root to tip), z wing normal.
    """

    mass: float  # kg
    r_com_WR: np.ndarray  # (3,) CoM position in WR coords, m
    I_com_WR: np.ndarray  # (3,3) inertia about the CoM, WR-aligned axes, kg m^2


def wing_mass_properties(wing: WingSpec) -> WingMassProperties | None:
    """Planform-derived mass properties; None when the wing is massless
    (``mass`` unset — the M1-M5 default, fully backward compatible)."""
    if wing.mass is None:
        return None
    m = wing.mass.value
    f_pa = wing.pitch_axis_fraction.value

    if wing.planform == "custom":
        # swept/tapered outline: x(s) varies along the span, so the CoM and
        # inertia need the x-weighted integrals directly (the f_pa shortcut
        # assumes x_c = (f_pa - 1/2) c, which sweep breaks)
        i_c1, i_sc, i_s2c, i_c3, i_xc, i_x2c, i_sxc = _custom_planform_integrals(wing)
        rho_a = m / i_c1
        s_com = i_sc / i_c1
        x_com = i_xc / i_c1
        i_xx = rho_a * (i_s2c - i_c1 * s_com**2)
        i_yy = rho_a * (i_x2c + i_c3 / 12.0) - m * x_com**2
        i_zz = i_xx + i_yy
        i_xy = m * x_com * s_com - rho_a * i_sxc
        return WingMassProperties(
            mass=m,
            r_com_WR=np.array([x_com, s_com, 0.0]),
            I_com_WR=np.array([[i_xx, i_xy, 0.0], [i_xy, i_yy, 0.0], [0.0, 0.0, i_zz]]),
        )

    i_c1, i_sc, i_s2c, i_c2, i_c3, i_sc2 = _planform_integrals(wing)
    rho_a = m / i_c1  # areal density, kg / m^2

    s_com = i_sc / i_c1
    x_com = (f_pa - 0.5) * i_c2 / i_c1  # element x_c = (f_pa - 0.5) * c(s)
    # thin plate, WR-aligned axes through the CoM:
    i_xx = rho_a * (i_s2c - i_c1 * s_com**2)  # int (s - s_com)^2 dm
    i_yy = rho_a * ((f_pa - 0.5) ** 2 + 1.0 / 12.0) * i_c3 - m * x_com**2
    i_zz = i_xx + i_yy  # perpendicular-axis theorem (z = 0 plate)
    # product of inertia I_xy = -int (x - x_com)(s - s_com) dm: nonzero when
    # the chordwise CoM offset varies along the span (taper + offset axis)
    i_xy = m * x_com * s_com - rho_a * (f_pa - 0.5) * i_sc2
    return WingMassProperties(
        mass=m,
        r_com_WR=np.array([x_com, s_com, 0.0]),
        I_com_WR=np.array([[i_xx, i_xy, 0.0], [i_xy, i_yy, 0.0], [0.0, 0.0, i_zz]]),
    )


# --- inertial terms at one instant --------------------------------------------

@dataclass
class WingInertialState:
    """Per-wing kinematic state needed to evaluate the inertial wrench after
    the dynamics solve (the full wrench depends on the solved base accel)."""

    mass: float
    I_cW: np.ndarray  # (3,3) wing CoM inertia in world coords
    c_W: np.ndarray  # (3,) wing CoM position, world
    r_rel_W: np.ndarray  # (3,) c_W - p_B (world)
    v_c_W: np.ndarray  # (3,) wing CoM velocity, world (velocity-level, complete)
    omega_w_W: np.ndarray  # (3,) wing angular velocity, world
    alpha_presc_W: np.ndarray  # (3,) wing angular accel with base accel zero
    a_c0_W: np.ndarray  # (3,) wing CoM accel with base accel zero
    R_wb: np.ndarray
    omega_B: np.ndarray  # base angular velocity, base coords
    v_B: np.ndarray  # base-origin velocity, base coords

    def wrench_on_thorax(self, alpha_b_B: np.ndarray, vdot_b_B: np.ndarray,
                         g_W: np.ndarray = G_W) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(F, N, c_W): inertial force (world, applied at the wing CoM) and
        moment about the wing CoM (world) transmitted to the thorax, i.e. what
        a force balance at the wing root would measure on top of the aero
        force. alpha_b_B / vdot_b_B are the SOLVED base angular acceleration
        and componentwise base-origin velocity derivative (base coords)."""
        alpha_b_W = self.R_wb @ np.asarray(alpha_b_B, dtype=float)
        a_b_W = self.R_wb @ (np.asarray(vdot_b_B, dtype=float) + cross3(self.omega_B, self.v_B))
        a_c = self.a_c0_W + a_b_W + cross3(alpha_b_W, self.r_rel_W)
        alpha_w = self.alpha_presc_W + alpha_b_W
        F = -self.mass * (a_c - g_W)
        N = -(self.I_cW @ alpha_w + cross3(self.omega_w_W, self.I_cW @ self.omega_w_W))
        return F, N, self.c_W


def wing_inertial_state(
    wing: WingSpec,
    kin: PrescribedWingKinematics,
    props: WingMassProperties,
    t: float,
    R_wb: np.ndarray,
    p_W: np.ndarray,
    omega_B: np.ndarray,
    v_B: np.ndarray,
) -> WingInertialState:
    """Wing CoM kinematics from the frozen transform chain (analytic rates)."""
    ks = wing_state(kin, t)
    R_WR, dR_WR, ddR_WR, omega_WR, alpha_WR = wing_chain_with_alpha(ks, wing.side)
    R_B_SP = R_B_SP_wing(kin.stroke_plane_angle.value, dihedral_of(wing), wing.side)
    R_world_SP = R_wb @ R_B_SP
    R_W_wing = R_world_SP @ R_WR

    r_c = props.r_com_WR
    r_rel_W = R_wb @ np.asarray(wing.hinge_location.value, float) + R_world_SP @ (R_WR @ r_c)
    v_flap = R_world_SP @ (dR_WR @ r_c)
    omega_b_W = R_wb @ omega_B
    omega_rel_W = R_world_SP @ (R_WR @ omega_WR)
    omega_w_W = omega_b_W + omega_rel_W
    # alpha with base angular accel zero: d(R_world_SP R_WR omega_WR)/dt with
    # dR_world_SP contributing omega_b x omega_rel (body rotation at fixed rate)
    alpha_presc_W = cross3(omega_b_W, omega_rel_W) \
        + R_world_SP @ (dR_WR @ omega_WR + R_WR @ alpha_WR)
    a_c0_W = (cross3(omega_b_W, cross3(omega_b_W, r_rel_W))
              + 2.0 * cross3(omega_b_W, v_flap)
              + R_world_SP @ (ddR_WR @ r_c))
    return WingInertialState(
        mass=props.mass,
        I_cW=R_W_wing @ props.I_com_WR @ R_W_wing.T,
        c_W=p_W + r_rel_W,
        r_rel_W=r_rel_W,
        v_c_W=R_wb @ v_B + cross3(omega_b_W, r_rel_W) + v_flap,
        omega_w_W=omega_w_W,
        alpha_presc_W=alpha_presc_W,
        a_c0_W=a_c0_W,
        R_wb=np.asarray(R_wb, dtype=float),
        omega_B=np.asarray(omega_B, dtype=float),
        v_B=np.asarray(v_B, dtype=float),
    )


def wing_inertial_terms(
    state: WingInertialState,
    g_W: np.ndarray = G_W,
) -> tuple[np.ndarray, np.ndarray]:
    """(M_add, Q_bias) for the model's base block (u = [omega_B; v_B; ...]).

    M_add: 6x6 spatial inertia of the wing about the base origin, base coords.
    Q_bias: 6-vector generalized wing-reaction wrench (velocity-level,
    prescribed-acceleration and gravity parts; the u_dot-dependent part is in
    M_add). Add to the mass matrix / right-hand side respectively (DR-006).
    """
    m = state.mass
    R_bw = state.R_wb.T
    r_B = R_bw @ state.r_rel_W
    I_cB = R_bw @ state.I_cW @ state.R_wb

    cx = skew(r_B)
    M_add = np.zeros((6, 6))
    M_add[:3, :3] = I_cB + m * (cx.T @ cx)
    M_add[:3, 3:] = m * cx
    M_add[3:, :3] = m * cx.T
    M_add[3:, 3:] = m * np.eye(3)

    # velocity/prescribed/gravity part of the wing reaction (world frame)
    F0 = -m * (state.a_c0_W + state.R_wb @ cross3(state.omega_B, state.v_B) - g_W)
    N0 = -(state.I_cW @ state.alpha_presc_W
           + cross3(state.omega_w_W, state.I_cW @ state.omega_w_W))
    Q_bias = np.zeros(6)
    Q_bias[:3] = R_bw @ (cross3(state.r_rel_W, F0) + N0)
    Q_bias[3:] = R_bw @ F0
    return M_add, Q_bias
