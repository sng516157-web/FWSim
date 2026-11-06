"""Blade-element discretization and element kinematic state (M1 Task 3).

Everything derives from the frozen transform chain (docs/conventions/frames.md):
positions, velocities and accelerations come from the analytic DCM chain and its
first two time derivatives — no finite differences. All aerodynamic models
consume the single ``ElementState`` representation (brief §5.2).

Assumptions: A-001 (rigid wing), A-004 (prescribed kinematics),
A-005 (no deformation).

Pitch axis: the wing rotates about a spanwise axis located at
``pitch_axis_fraction`` of the local chord measured from the leading edge
(0 = leading edge, 0.5 = mid-chord). The BE origin is the element's mid-chord
centre; the chordwise offset enters as x_c = (pitch_axis_fraction - 0.5) * c
along +x_WR (which points toward the leading edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from fwmav_sim.core.transforms import R_W_B, R_B_SP_wing, Rx, Ry, Rz, cross3, skew
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics, WingKinematicState, wing_state
from fwmav_sim.vehicle.schema import WingSpec

# --- planform geometry --------------------------------------------------------

def dihedral_of(wing: WingSpec) -> float:
    """Per-wing stroke-plane dihedral gamma [rad] (DR-005; 0 if unset)."""
    return wing.dihedral_angle.value if wing.dihedral_angle is not None else 0.0


def chord_at(wing: WingSpec, s: float) -> float:
    """Local chord c(s) at span station s for the configured planform."""
    R = wing.span.value
    c_r, c_t = wing.root_chord.value, wing.tip_chord.value
    u = np.clip(s / R, 0.0, 1.0)
    if wing.planform == "rectangular":
        return c_r
    if wing.planform == "trapezoidal":
        return c_r + (c_t - c_r) * u
    if wing.planform == "elliptical":
        return c_r * np.sqrt(max(1.0 - u * u, 0.0))
    if wing.planform == "custom":
        st = wing.planform_stations
        return float(np.interp(s, [p.y for p in st], [p.chord for p in st]))
    raise ValueError(f"unknown planform {wing.planform!r}")


def le_offset_at(wing: WingSpec, s: float) -> float:
    """Leading-edge x position (x_WR, toward LE) at station s. Zero for the
    analytic planforms (they are unswept); interpolated from the stations
    for custom. For custom planforms the pitch axis is the straight x=0 line:
    le_offset carries the sweep AND the spar position (pitch_axis_fraction
    is ignored — set it implicitly via le_offset = f * chord for an unswept
    outline)."""
    if wing.planform == "custom":
        st = wing.planform_stations
        return float(np.interp(s, [p.y for p in st], [p.le_offset for p in st]))
    return wing.pitch_axis_fraction.value * chord_at(wing, s)


def cumulative_area(wing: WingSpec, s: float) -> float:
    """Planform area between the root and span station s (analytic; exact
    trapezoids for the piecewise-linear custom outline)."""
    R = wing.span.value
    c_r, c_t = wing.root_chord.value, wing.tip_chord.value
    u = np.clip(s / R, 0.0, 1.0)
    if wing.planform == "rectangular":
        return c_r * R * u
    if wing.planform == "trapezoidal":
        return R * (c_r * u + 0.5 * (c_t - c_r) * u * u)
    if wing.planform == "elliptical":
        return 0.5 * c_r * R * (u * np.sqrt(max(1.0 - u * u, 0.0)) + np.arcsin(u))
    if wing.planform == "custom":
        st = wing.planform_stations
        s = min(max(s, 0.0), R)
        area = 0.0
        for a, b in zip(st, st[1:]):
            if s <= a.y:
                break
            hi = min(s, b.y)
            frac = (hi - a.y) / (b.y - a.y)
            c_hi = a.chord + frac * (b.chord - a.chord)
            area += 0.5 * (a.chord + c_hi) * (hi - a.y)
            if s <= b.y:
                break
        return area
    raise ValueError(f"unknown planform {wing.planform!r}")


def planform_area(wing: WingSpec) -> float:
    """Total single-wing planform area."""
    return cumulative_area(wing, wing.span.value)


@dataclass(frozen=True)
class BladeElement:
    """Geometry of one blade element."""

    s: float  # spanwise station of the element centre, m (along +y_WR)
    chord: float  # area-equivalent local chord, m
    area: float  # element planform area, m^2
    x_c: float  # chordwise offset of the element centre from the pitch axis, m


def discretize(
    wing: WingSpec, n_elements: int, method: Literal["equal_span", "equal_area"] = "equal_span"
) -> list[BladeElement]:
    """Split the span into n blade elements.

    Element areas always integrate exactly: each element's area is the
    analytic planform area between its edges, and its chord is the
    area-equivalent chord (area / element width).
    """
    R = wing.span.value
    if method == "equal_span":
        edges = np.linspace(0.0, R, n_elements + 1)
    elif method == "equal_area":
        total = planform_area(wing)
        inner = [
            brentq(lambda s, k=k: cumulative_area(wing, s) - k * total / n_elements, 0.0, R)
            for k in range(1, n_elements)
        ]
        edges = np.asarray([0.0, *inner, R])
    else:
        raise ValueError(f"unknown discretization {method!r}")
    elements = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        area = cumulative_area(wing, b) - cumulative_area(wing, a)
        chord = area / (b - a)
        s_mid = 0.5 * (a + b)
        # element centre (mid-chord) x in WR. Legacy form for the analytic
        # planforms (area-equivalent chord, unchanged numerics); for custom
        # outlines the le_offset carries sweep + spar position.
        if wing.planform == "custom":
            x_c = le_offset_at(wing, s_mid) - 0.5 * chord
        else:
            x_c = (wing.pitch_axis_fraction.value - 0.5) * chord
        elements.append(BladeElement(s=s_mid, chord=chord, area=area, x_c=x_c))
    return elements


# --- wing DCM chain with analytic rates ---------------------------------------

_AXIS = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}
_ELEM = {"x": Rx, "y": Ry, "z": Rz}


def _chain_with_rates(
    terms: list[tuple[str, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DCM R = R1 R2 ... Rn, its first two time derivatives, and angular
    velocity/acceleration in the child frame.

    Each term is (axis, angle, rate, rrate) for an elementary rotation
    R_i = R_axis(angle). Returns (R, dR, ddR, omega_body, alpha_body) where
    omega_body / alpha_body are the chain's angular velocity / componentwise
    angular acceleration expressed in the final (child) frame.
    Uses dRi = Ri [axis]x * rate, ddRi = Ri ([axis]x^2 rate^2 + [axis]x rrate).
    alpha_body = d(omega_body)/dt = sum_i [ d(S_i^T)/dt axis_i rate_i
    + S_i^T axis_i rrate_i ] with S_i = R_{i+1}..R_n and
    dS_i/dt = sum_{j>i} R_{i+1}..dR_j..R_n (needed by the wing-inertia terms,
    dynamics/wing_inertia.py; verified by finite difference in the tests).
    """
    n = len(terms)
    Rs, dRs, ddRs = [], [], []
    for axis, angle, rate, rrate in terms:
        ax = _AXIS[axis]
        Ri = _ELEM[axis](angle)
        sx = skew(ax)
        Rs.append(Ri)
        dRs.append(Ri @ sx * rate)
        ddRs.append(Ri @ (sx @ sx * rate * rate + sx * rrate))

    # prefix products pre[i] = R_0..R_{i-1} and suffix suf[i] = R_i..R_{n-1}
    pre = [np.eye(3)]
    for i in range(n):
        pre.append(pre[i] @ Rs[i])
    suf = [np.eye(3)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = Rs[i] @ suf[i + 1]
    # dsuf[i] = d(suf[i])/dt
    dsuf = [np.zeros((3, 3)) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        acc = np.zeros((3, 3))
        mid = np.eye(3)
        for j in range(i, n):
            acc += mid @ dRs[j] @ suf[j + 1]
            mid = mid @ Rs[j]
        dsuf[i] = acc

    R = pre[n]
    dR = np.zeros((3, 3))
    ddR = np.zeros((3, 3))
    for i in range(n):
        dR += pre[i] @ dRs[i] @ suf[i + 1]
        ddR += pre[i] @ ddRs[i] @ suf[i + 1]
        for j in range(i + 1, n):
            mid = np.eye(3)
            for k in range(i + 1, j):
                mid = mid @ Rs[k]
            ddR += 2.0 * pre[i] @ dRs[i] @ mid @ dRs[j] @ suf[j + 1]
    # angular velocity in the child frame: sum_i S_i^T axis_i rate_i
    omega = np.zeros(3)
    alpha = np.zeros(3)
    for i in range(n):
        ax = _AXIS[terms[i][0]]
        omega += suf[i + 1].T @ (ax * terms[i][2])
        alpha += dsuf[i + 1].T @ (ax * terms[i][2]) + suf[i + 1].T @ (ax * terms[i][3])
    return R, dR, ddR, omega, alpha


def wing_chain_with_alpha(
    kin_state: WingKinematicState, side: Literal["right", "left"] = "right"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """R_SP_WR with its first two derivatives plus omega and alpha in WR
    (frames.md §3-4). alpha is the componentwise child-frame angular
    acceleration (chain derivative, not the rigid-body alpha vector)."""
    s = kin_state
    if side == "right":
        terms = [
            ("z", -s.phi, -s.phi_dot, -s.phi_ddot),
            ("x", s.psi_w, s.psi_w_dot, s.psi_w_ddot),
            ("y", s.theta_w, s.theta_w_dot, s.theta_w_ddot),
        ]
        return _chain_with_rates(terms)
    if side == "left":
        terms = [
            ("z", s.phi, s.phi_dot, s.phi_ddot),
            ("x", -s.psi_w, -s.psi_w_dot, -s.psi_w_ddot),
            ("y", s.theta_w, s.theta_w_dot, s.theta_w_ddot),
        ]
        R, dR, ddR, omega, alpha = _chain_with_rates(terms)
        R0 = Rx(np.pi)  # constant left-wing base orientation (frames.md §4)
        return R @ R0, dR @ R0, ddR @ R0, R0.T @ omega, R0.T @ alpha
    raise ValueError(f"side must be 'right' or 'left', got {side!r}")


def wing_chain(
    kin_state: WingKinematicState, side: Literal["right", "left"] = "right"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """R_SP_WR with its first two derivatives and omega in WR (frames.md §3-4)."""
    R, dR, ddR, omega, _ = wing_chain_with_alpha(kin_state, side)
    return R, dR, ddR, omega


# --- element state --------------------------------------------------------------

@dataclass
class ElementState:
    """Single kinematic-state representation consumed by all aero models (brief §5.2).

    Vectors are expressed in the world frame W (per the M1 stage-file
    dataclass); BE axes are available via R_W_BE.
    """

    r_W: np.ndarray  # (3,) element position, world frame
    v_W: np.ndarray  # (3,) element velocity, world frame
    a_W: np.ndarray  # (3,) element acceleration, world frame
    R_W_BE: np.ndarray  # (3,3) element orientation (maps BE coords -> W)
    omega_W: np.ndarray  # (3,) wing angular velocity, world frame
    chord: float  # m
    area: float  # m^2
    alpha: float  # local geometric angle of attack, rad (frames.md §3/§4 sign rule)
    reynolds: float  # Re = |v_rel| c / nu
    side: Literal["right", "left"]
    # M1 extensions needed by the rotational/added-mass terms (stage-file
    # dataclass extended, documented in the M1 report):
    pitch_rate: float  # theta_w_dot [rad/s]; identical command for both wings
    pitch_axis_fraction: float  # x_hat_0: pitch axis location, 0 = leading edge
    t: float = 0.0  # element evaluation time (needed by history-aware terms, M6)
    s: float = 0.0  # spanwise station of the element centre (unsteady terms)
    aux: dict | None = None  # optional runner-provided context (unsteady terms)


class WingFrameContext:
    """Per-(wing, t) shared kinematic quantities for element evaluation.

    The frozen transform chain and its rates are identical for all elements
    of a wing; computing them once per wing per call (not per element) cuts
    the per-element cost by the element count.
    """

    __slots__ = ("ks", "R_WR", "dR_WR", "ddR_WR", "omega_WR", "R_world_SP",
                 "R_wb", "hinge_W", "omega_W", "v_B_W", "v_air_W", "p_B_W", "aux")

    def __init__(self, wing, kin, t, R_wb, p_B_W, omega_W, v_B_W, v_air_W):
        self.aux = None  # runner-provided context for optional terms (M6)
        self.ks = wing_state(kin, t)
        self.R_WR, self.dR_WR, self.ddR_WR, self.omega_WR = wing_chain(self.ks, wing.side)
        self.R_world_SP = R_wb @ R_B_SP_wing(kin.stroke_plane_angle.value,
                                             dihedral_of(wing), wing.side)
        self.R_wb = R_wb
        self.hinge_W = R_wb @ np.asarray(wing.hinge_location.value, dtype=float)
        self.p_B_W = np.asarray(p_B_W)
        self.omega_W = np.asarray(omega_W)
        self.v_B_W = np.asarray(v_B_W)
        self.v_air_W = np.zeros(3) if v_air_W is None else np.asarray(v_air_W, dtype=float)


def batched_element_state_ctx(wing: WingSpec, elements: list[BladeElement],
                              ctx: WingFrameContext, nu: float) -> dict[str, np.ndarray]:
    """Vectorized element states for one wing (M6.75 perf pass): identical
    math to element_state_ctx applied to all elements at once (arrays over
    elements). Keys: r_W, v_W, a_W (n,3); alpha, reynolds (n,); plus wing-level
    scalars/vectors shared by the force evaluation."""
    n = len(elements)
    r_BE = np.array([[el.x_c, el.s, 0.0] for el in elements])  # (n,3)
    chord = np.array([el.chord for el in elements])
    area = np.array([el.area for el in elements])

    r_rel_B = (ctx.R_wb @ np.asarray(wing.hinge_location.value, float))[None, :] \
        + (ctx.R_world_SP @ ctx.R_WR @ r_BE.T).T
    r_W = ctx.p_B_W + r_rel_B
    R_W_BE = ctx.R_world_SP @ ctx.R_WR
    v_flap = (ctx.R_world_SP @ ctx.dR_WR @ r_BE.T).T
    v_W = ctx.v_B_W[None, :] + np.cross(ctx.omega_W, r_rel_B) + v_flap
    a_W = (np.cross(ctx.omega_W, np.cross(ctx.omega_W, r_rel_B))
           + 2.0 * np.cross(ctx.omega_W, v_flap)
           + (ctx.R_world_SP @ ctx.ddR_WR @ r_BE.T).T)
    v_rel = v_W - ctx.v_air_W
    chord_dir = R_W_BE @ np.array([1.0, 0.0, 0.0])
    n_phys = R_W_BE @ np.array([0.0, 0.0, 1.0 if wing.side == "right" else -1.0])
    alpha = np.arctan2(v_rel @ n_phys, v_rel @ chord_dir)
    speed = np.linalg.norm(v_rel, axis=1)
    return {
        "r_W": r_W, "v_W": v_W, "a_W": a_W, "alpha": alpha,
        "reynolds": speed * chord / nu, "chord": chord, "area": area,
        "chord_dir": chord_dir, "n_phys": n_phys, "v_rel": v_rel,
        "speed": speed, "pitch_rate": ctx.ks.theta_w_dot,
        "pitch_axis_fraction": wing.pitch_axis_fraction.value,
    }


def batched_element_state(wing: WingSpec, elements: list[BladeElement],
                          kin: PrescribedWingKinematics, t: float, nu: float,
                          v_air_W: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Fixed-body batch (body at identity pose, at rest)."""
    ctx = WingFrameContext(wing, kin, t, np.eye(3), np.zeros(3), np.zeros(3),
                           np.zeros(3), v_air_W)
    return batched_element_state_ctx(wing, elements, ctx, nu)


def element_state_ctx(wing: WingSpec, element: BladeElement, ctx: WingFrameContext,
                      nu: float) -> ElementState:
    """Element state from a shared per-wing context (see element_state_free_body)."""
    side: Literal["right", "left"] = wing.side
    r_BE = np.array([element.x_c, element.s, 0.0])
    r_rel_B = ctx.R_wb @ np.asarray(wing.hinge_location.value, float) \
        + ctx.R_world_SP @ (ctx.R_WR @ r_BE)
    r_W = ctx.p_B_W + r_rel_B
    R_W_BE = ctx.R_world_SP @ ctx.R_WR
    v_flap = ctx.R_world_SP @ (ctx.dR_WR @ r_BE)
    v_W = ctx.v_B_W + cross3(ctx.omega_W, r_rel_B) + v_flap
    a_W = (cross3(ctx.omega_W, cross3(ctx.omega_W, r_rel_B))
           + 2.0 * cross3(ctx.omega_W, v_flap)
           + ctx.R_world_SP @ (ctx.ddR_WR @ r_BE))
    omega_el_W = ctx.omega_W + ctx.R_world_SP @ (ctx.R_WR @ ctx.omega_WR)
    v_rel = v_W - ctx.v_air_W
    chord_dir = R_W_BE @ np.array([1.0, 0.0, 0.0])
    n_phys = R_W_BE @ np.array([0.0, 0.0, 1.0 if side == "right" else -1.0])
    alpha = float(np.arctan2(v_rel @ n_phys, v_rel @ chord_dir))
    speed = float(np.linalg.norm(v_rel))
    return ElementState(
        r_W=r_W, v_W=v_W, a_W=a_W, R_W_BE=R_W_BE, omega_W=omega_el_W,
        chord=element.chord, area=element.area, alpha=alpha,
        reynolds=speed * element.chord / nu, side=side,
        pitch_rate=ctx.ks.theta_w_dot,
        pitch_axis_fraction=wing.pitch_axis_fraction.value,
        t=ctx.ks.t,
        s=element.s,
        aux=ctx.aux,
    )


def element_state_free_body(
    wing: WingSpec,
    element: BladeElement,
    kin: PrescribedWingKinematics,
    t: float,
    nu: float,
    R_wb: np.ndarray,
    p_B_W: np.ndarray,
    omega_W: np.ndarray,
    v_B_W: np.ndarray,
    v_air_W: np.ndarray | None = None,
) -> ElementState:
    """Element kinematic state for a MOVING body (M3 Task 4) — convenience
    wrapper building a fresh WingFrameContext; bulk callers should build the
    context once per wing per timestep and call element_state_ctx."""
    ctx = WingFrameContext(wing, kin, t, R_wb, p_B_W, omega_W, v_B_W, v_air_W)
    return element_state_ctx(wing, element, ctx, nu)


def element_state(
    wing: WingSpec,
    element: BladeElement,
    kin: PrescribedWingKinematics,
    t: float,
    nu: float,
    v_air_W: np.ndarray | None = None,
    body_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_position_W: np.ndarray | None = None,
) -> ElementState:
    """Kinematic state of one blade element of a *fixed* body (M1 scope).

    The body does not move (M1 ground rule); ``body_euler``/``body_position_W``
    allow a fixed non-identity pose. Relative airflow uses
    v_rel = v_element - v_air (brief §5.4, assumption A-006).
    """
    side: Literal["right", "left"] = wing.side
    ks: WingKinematicState = wing_state(kin, t)
    R_WR, dR_WR, ddR_WR, omega_WR = wing_chain(ks, side)
    R_world_SP = R_W_B(*body_euler) @ R_B_SP_wing(kin.stroke_plane_angle.value,
                                                  dihedral_of(wing), wing.side)

    r_BE = np.array([element.x_c, element.s, 0.0])  # in WR coords
    hinge = np.asarray(wing.hinge_location.value, dtype=float)  # in B (frames.md §6)
    origin_W = (np.zeros(3) if body_position_W is None else np.asarray(body_position_W)) + R_W_B(
        *body_euler
    ) @ hinge

    R_W_BE = R_world_SP @ R_WR
    r_W = origin_W + R_world_SP @ (R_WR @ r_BE)
    v_W = R_world_SP @ (dR_WR @ r_BE)
    a_W = R_world_SP @ (ddR_WR @ r_BE)
    omega_W = R_world_SP @ (R_WR @ omega_WR)

    if v_air_W is None:
        v_air_W = np.zeros(3)
    v_rel = v_W - np.asarray(v_air_W, dtype=float)

    # Geometric angle of attack (flight-dynamics convention, frames.md §3/§4):
    # alpha = atan2(w, u) with (u, w) the element's velocity relative to the
    # air in the BE chord-normal plane; the spanwise component is excluded.
    # Positive theta_w (leading edge up) increases alpha for +x motion.
    # n_phys = +z_BE (right) / -z_BE (left).
    chord_dir = R_W_BE @ np.array([1.0, 0.0, 0.0])
    n_phys = R_W_BE @ np.array([0.0, 0.0, 1.0 if side == "right" else -1.0])
    alpha = float(np.arctan2(v_rel @ n_phys, v_rel @ chord_dir))

    speed = float(np.linalg.norm(v_rel))
    reynolds = speed * element.chord / nu

    return ElementState(
        r_W=r_W,
        v_W=v_W,
        a_W=a_W,
        R_W_BE=R_W_BE,
        omega_W=omega_W,
        chord=element.chord,
        area=element.area,
        alpha=alpha,
        reynolds=reynolds,
        side=side,
        pitch_rate=ks.theta_w_dot,
        pitch_axis_fraction=wing.pitch_axis_fraction.value,
        t=t,
        s=element.s,
    )
