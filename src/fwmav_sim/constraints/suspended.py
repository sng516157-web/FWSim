"""Suspended-string pendulum constraint model (M2 Task 5).

A suspended-string experiment is NOT free flight (brief §6.3): the vehicle
pivots about a fixed point. This model is a deliberately separate plant from
the free reduced model (dynamics/reduced_longitudinal.py).

Modelling decision (documented in the M2 report): the string is modelled as a
**rigid massless pivot** — the vehicle rotates about the suspension point with
no free rotation at the attachment and no string slack. This is exact for the
small-oscillation swing mode tested here (compound-pendulum frequency
omega = sqrt(m g d_cm / I_pivot)) and matches the level of the analytical
reference; a string with a free attach-point rotation would be a different
plant and is deferred.

Generalized coordinates: q_s = [phi_p, alpha_abd]
  phi_p — swing angle about the pivot (+y axis), phi_p = 0 = body level
          hanging below the pivot; positive = nose-up rotation.
  alpha_abd — abdomen pitch relative to thorax (frozen sign: + = tip down).

Body geometry in B: thorax CoM offset from pivot u_t = (u_tx, u_tz), abdomen
joint offset from pivot u_J = (j_x, j_z), abdomen CoM l_a behind the joint
along the body x-axis. All derivations cross-checked against the SymPy
reference (_reduced_reference.pend_*) to < 1e-10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha


@dataclass(frozen=True)
class SuspendedPendulumParams:
    m_t: float
    m_a: float
    I_t: float  # thorax pitch inertia about its own CoM
    u_tx: float  # thorax CoM offset from pivot, B x
    u_tz: float  # thorax CoM offset from pivot, B z (positive = below pivot)
    j_x: float  # abdomen joint offset from pivot, B x
    j_z: float  # abdomen joint offset from pivot, B z
    l_a: float  # joint-to-abdomen-CoM distance along the abdomen axis
    I_a: float = 0.0  # abdomen rotary inertia about its CoM (spec §1 extension)
    g: float = 9.81
    c_a: float = 0.0  # abdomen joint damping, N m s / rad
    lock_abdomen: bool = False

    @classmethod
    def from_vehicle(cls, vehicle, pivot_body_point: tuple[float, float], **kw):
        """Build from a VehicleDefinition: pivot at a body-frame point.

        pivot_body_point = (x, z) in B. Walks the serial chain thorax ->
        pitch joint -> (roll carrier) -> abdomen; the abdomen CoM must lie on
        the body x-axis behind the pitch joint (checked). Frame semantics
        (M3): child body frame origin = joint location; com_offset is in the
        body frame itself.
        """
        bodies = {b.name: b for b in vehicle.bodies}
        pitch = next(j for j in vehicle.joints if j.name == "abdomen_pitch")
        thorax = bodies[pitch.parent_body]
        # terminal body of the chain
        child_of = {j.parent_body: j for j in vehicle.joints}
        terminal = pitch.child_body
        while terminal in child_of:
            terminal = child_of[terminal].child_body
        abdomen = bodies[terminal]
        r_t = np.asarray(thorax.com_offset.value)
        w = np.asarray(abdomen.com_offset.value)  # in the abdomen frame (origin at joint)
        r_J = np.asarray(pitch.location.value)
        if abs(w[1]) > 1e-15 or abs(w[2]) > 1e-15 or w[0] >= 0:
            raise ValueError("abdomen CoM must lie on the body x-axis behind the joint")
        return cls(
            m_t=thorax.mass.value, m_a=abdomen.mass.value,
            I_t=float(np.asarray(thorax.inertia.value)[1, 1]),
            u_tx=float(r_t[0] - pivot_body_point[0]),
            u_tz=float(r_t[2] - pivot_body_point[1]),
            j_x=float(r_J[0] - pivot_body_point[0]),
            j_z=float(r_J[2] - pivot_body_point[1]),
            l_a=float(-w[0]),
            **kw,
        )


def _K(params: SuspendedPendulumParams, al: float) -> tuple[float, float]:
    """K(alpha) = u_J . Ry(alpha) w and dK/dalpha, w = (-l_a, 0)."""
    l_a, jx, jz = params.l_a, params.j_x, params.j_z
    K = l_a * (-jx * np.cos(al) + jz * np.sin(al))
    dK = l_a * (jx * np.sin(al) + jz * np.cos(al))
    return K, dK


def pend_mass_matrix(q_s: np.ndarray, p: SuspendedPendulumParams) -> np.ndarray:
    _, al = q_s
    K, _ = _K(p, al)
    m_phiphi = (p.m_t * (p.u_tx**2 + p.u_tz**2) + p.I_t + p.I_a
                + p.m_a * (p.j_x**2 + p.j_z**2) + p.m_a * p.l_a**2 + 2 * p.m_a * K)
    m_pa = p.m_a * (p.l_a**2 + K) + p.I_a
    return np.array([[m_phiphi, m_pa], [m_pa, p.m_a * p.l_a**2 + p.I_a]])


def pend_bias(q_s: np.ndarray, qd_s: np.ndarray, p: SuspendedPendulumParams) -> np.ndarray:
    _, al = q_s
    phid, ald = qd_s
    _, dK = _K(p, al)
    return np.array([2 * p.m_a * dK * phid * ald + p.m_a * dK * ald**2,
                     -p.m_a * dK * phid**2])


def pend_gravity(q_s: np.ndarray, p: SuspendedPendulumParams) -> np.ndarray:
    phi, al = q_s
    sph, cph = np.sin(phi), np.cos(phi)
    cpa = np.cos(phi + al)
    G_phi = -p.g * (p.m_t * (-cph * p.u_tx - sph * p.u_tz)
                    + p.m_a * (-cph * p.j_x - sph * p.j_z + p.l_a * cpa))
    G_alpha = -p.g * p.m_a * p.l_a * cpa
    return np.array([G_phi, G_alpha])


class SuspendedPendulum:
    """Rigid-pivot suspended vehicle with free or prescribed abdomen."""

    def __init__(self, params: SuspendedPendulumParams):
        self.p = params
        self.last_tau_act_required: float = 0.0

    def rhs(
        self,
        state: np.ndarray,
        t: float,
        tau_act: float = 0.0,
        alpha_profile: AbdomenProfile | None = None,
        alpha_limits: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """state = [phi_p, alpha_abd, phi_p_dot, alpha_abd_dot]."""
        p = self.p
        q_s = np.asarray(state[:2], dtype=float)
        qd_s = np.asarray(state[2:], dtype=float)
        if alpha_profile is not None:
            a, ad, add = prescribed_alpha(alpha_profile, t, alpha_limits)
            q_s[1], qd_s[1] = a, ad
        M = pend_mass_matrix(q_s, p)
        rhs = -pend_bias(q_s, qd_s, p) - pend_gravity(q_s, p)
        rhs[1] += tau_act - p.c_a * qd_s[1]

        qdd = np.zeros(2)
        if alpha_profile is not None:
            qdd[0] = (rhs[0] - M[0, 1] * add) / M[0, 0]
            qdd[1] = add
            self.last_tau_act_required = float(M[1, :] @ qdd - rhs[1] + tau_act)
        elif p.lock_abdomen:
            qdd[0] = rhs[0] / M[0, 0]
        else:
            qdd = np.linalg.solve(M, rhs)
            self.last_tau_act_required = tau_act
        return np.concatenate([qd_s, qdd])

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy (kinetic + potential) of the assembly."""
        p = self.p
        phi, al = state[0], state[1]
        phid, ald = state[2], state[3]
        sph, cph = np.sin(phi), np.cos(phi)
        spa, cpa = np.sin(phi + al), np.cos(phi + al)
        # velocities of the two point masses (pivot at origin of measure)
        vt = phid * np.array([-sph * p.u_tx + cph * p.u_tz, -cph * p.u_tx - sph * p.u_tz])
        vj = phid * np.array([-sph * p.j_x + cph * p.j_z, -cph * p.j_x - sph * p.j_z])
        va = vj + (phid + ald) * np.array([p.l_a * spa, p.l_a * cpa])
        T = (0.5 * p.m_t * float(vt @ vt) + 0.5 * p.I_t * phid**2
             + 0.5 * p.I_a * (phid + ald) ** 2 + 0.5 * p.m_a * float(va @ va))
        z_t = -sph * p.u_tx + cph * p.u_tz
        z_a = -sph * p.j_x + cph * p.j_z + p.l_a * spa
        V = -(p.m_t * z_t + p.m_a * z_a) * p.g
        return float(T + V)


def pendulum_small_oscillation_frequency(p: SuspendedPendulumParams) -> float:
    """Analytical omega = sqrt(m g d_cm / I_pivot) for the locked, level
    equilibrium (pivot directly above the CoM at alpha_abd = 0)."""
    m = p.m_t + p.m_a
    # CoM offset from pivot at alpha = 0
    x_com = (p.m_t * p.u_tx + p.m_a * (p.j_x - p.l_a)) / m
    z_com = (p.m_t * p.u_tz + p.m_a * p.j_z) / m
    d_cm = float(np.hypot(x_com, z_com))
    I_pivot = (p.I_t + p.I_a + p.m_t * (p.u_tx**2 + p.u_tz**2)
               + p.m_a * ((p.j_x - p.l_a) ** 2 + p.j_z**2))
    return float(np.sqrt(m * p.g * d_cm / I_pivot))
