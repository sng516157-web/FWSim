"""NumPy runtime reduced longitudinal thorax-abdomen model (M2 Task 4).

Implements docs/conventions/reduced_model_spec.md:
  state x = [x, z, theta, alpha_abd, x_dot, z_dot, theta_dot, alpha_abd_dot]
  M(q) qdd + bias(q, qd) + G(q) = Q(forces, tau_act, damping)

Hand-written; cross-checked against the SymPy-derived reference
(dynamics/_reduced_reference.py) at 100 random states to < 1e-10
(tests/analytical/test_reduced_model.py). SymPy-free at runtime (M2 §2.2).

Force interface: point forces given in the BODY frame B, applied at body
points (b_x, b_z) — cycle-averaged wing forces and the foil's synthetic
thrust both enter this way. Abdomen handling:
  - free: alpha_abd is a state; tau_act applies to Q_alpha (relative-angle
    coordinate; no explicit Q_theta reaction term — see spec §5)
  - prescribed: alpha_abd(t) from an AbdomenProfile overrides the state;
    the body 3x3 block is solved with the coupling terms and the required
    actuator torque is returned (brief §6.4 stage 1: record reaction torque)
  - locked: alpha fixed at its initial value, alpha_dot = alpha_ddot = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha
from fwmav_sim.dynamics import _reduced_reference as ref

PointForce = tuple[tuple[float, float], tuple[float, float]]  # ((bx, bz), (Fx_B, Fz_B))


@dataclass(frozen=True)
class ReducedModelParams:
    m_t: float  # thorax mass, kg
    m_a: float  # abdomen mass, kg (0 allowed: coincident-mass foil limit)
    I_t: float  # thorax pitch inertia about its CoM, kg m^2
    d: float  # abdomen joint distance behind the thorax reference point, m
    l_a: float  # joint-to-abdomen-CoM distance, m
    g: float = 9.81
    c_lin: float = 0.0  # linear damping on x_dot, z_dot, N s / m
    c_ang: float = 0.0  # angular damping on theta_dot, N m s / rad
    c_a: float = 0.0  # abdomen joint damping, N m s / rad
    lock_abdomen: bool = False

    @classmethod
    def from_vehicle(cls, vehicle, **kw) -> ReducedModelParams:
        """Map a VehicleDefinition (thorax + abdomen via one revolute pitch
        joint) onto the reduced-model parameters (reduced_model_spec.md §1-2)."""
        bodies = {b.name: b for b in vehicle.bodies}
        joint = next(j for j in vehicle.joints if j.type == "revolute")
        thorax = bodies[joint.parent_body]
        abdomen = bodies[joint.child_body]
        r_J = np.asarray(joint.location.value, dtype=float)
        w = np.asarray(abdomen.com_offset.value, dtype=float) - r_J
        if abs(w[1]) > 1e-15 or abs(w[2]) > 1e-15 or w[0] >= 0:
            raise ValueError("abdomen CoM must lie on the body x-axis behind the joint")
        return cls(
            m_t=thorax.mass.value,
            m_a=abdomen.mass.value,
            I_t=float(np.asarray(thorax.inertia.value, dtype=float)[1, 1]),
            d=float(-r_J[0]),  # joint behind the thorax reference point => r_Jx < 0
            l_a=float(-w[0]),
            **kw,
        )


def mass_matrix(q: np.ndarray, p: ReducedModelParams) -> np.ndarray:
    """M(q), q = [x, z, theta, alpha_abd]. Hand-derived; matches ref.free_M."""
    _, _, th, al = q
    m_a, d, l_a = p.m_a, p.d, p.l_a
    sth, cth = np.sin(th), np.cos(th)
    stha, ctha = np.sin(th + al), np.cos(th + al)
    return np.array([
        [p.m_t + m_a, 0.0, m_a * (d * sth + l_a * stha), m_a * l_a * stha],
        [0.0, p.m_t + m_a, m_a * (d * cth + l_a * ctha), m_a * l_a * ctha],
        [m_a * (d * sth + l_a * stha), m_a * (d * cth + l_a * ctha),
         p.I_t + m_a * (d * d + l_a * l_a + 2 * d * l_a * np.cos(al)),
         m_a * (l_a * l_a + d * l_a * np.cos(al))],
        [m_a * l_a * stha, m_a * l_a * ctha, m_a * (l_a * l_a + d * l_a * np.cos(al)),
         m_a * l_a * l_a],
    ])


def _dM(q: np.ndarray, p: ReducedModelParams) -> tuple[np.ndarray, np.ndarray]:
    """dM/dtheta, dM/dalpha (M depends only on theta, alpha)."""
    _, _, th, al = q
    m_a, d, l_a = p.m_a, p.d, p.l_a
    sth, cth = np.sin(th), np.cos(th)
    stha, ctha = np.sin(th + al), np.cos(th + al)
    dM_th = np.zeros((4, 4))
    dM_th[0, 2] = dM_th[2, 0] = m_a * (d * cth + l_a * ctha)
    dM_th[0, 3] = dM_th[3, 0] = m_a * l_a * ctha
    dM_th[1, 2] = dM_th[2, 1] = -m_a * (d * sth + l_a * stha)
    dM_th[1, 3] = dM_th[3, 1] = -m_a * l_a * stha
    dM_al = np.zeros((4, 4))
    dM_al[0, 2] = dM_al[2, 0] = m_a * l_a * ctha
    dM_al[0, 3] = dM_al[3, 0] = m_a * l_a * ctha
    dM_al[1, 2] = dM_al[2, 1] = -m_a * l_a * stha
    dM_al[1, 3] = dM_al[3, 1] = -m_a * l_a * stha
    dM_al[2, 2] = -2.0 * m_a * d * l_a * np.sin(al)
    dM_al[2, 3] = dM_al[3, 2] = -m_a * d * l_a * np.sin(al)
    return dM_th, dM_al


def bias_vector(q: np.ndarray, qd: np.ndarray, p: ReducedModelParams) -> np.ndarray:
    """Coriolis/centrifugal vector via Christoffel contraction of dM/dq."""
    dM = [np.zeros((4, 4)), np.zeros((4, 4)), *_dM(q, p)]
    bias = np.zeros(4)
    for i in range(4):
        # sum_jk dM_ij/dq_k qd_j qd_k - 1/2 sum_jk dM_jk/dq_i qd_j qd_k
        for k in (2, 3):
            bias[i] += qd[k] * float(dM[k][i, :] @ qd)
        bias[i] -= 0.5 * float(qd @ dM[i] @ qd)
    return bias


def gravity_vector(q: np.ndarray, p: ReducedModelParams) -> np.ndarray:
    """G(q) = dV/dq, V = -(m_t z + m_a z_a) g (z down)."""
    _, _, th, al = q
    m_a, d, l_a = p.m_a, p.d, p.l_a
    return np.array([
        0.0,
        -(p.m_t + m_a) * p.g,
        -m_a * p.g * (d * np.cos(th) + l_a * np.cos(th + al)),
        -m_a * p.g * l_a * np.cos(th + al),
    ])


def generalized_force(theta: float, forces: list[PointForce]) -> np.ndarray:
    """Q from body-frame point forces: Q_i = F_W . d r_P/d q_i."""
    sth, cth = np.sin(theta), np.cos(theta)
    Q = np.zeros(4)
    for (bx, bz), (fx_b, fz_b) in forces:
        fx_w = cth * fx_b + sth * fz_b  # Ry(theta) @ (fx_b, fz_b)
        fz_w = -sth * fx_b + cth * fz_b
        Q[0] += fx_w
        Q[1] += fz_w
        # d r_P/d theta = Ry'(theta) @ (bx, bz)
        Q[2] += fx_w * (-sth * bx + cth * bz) + fz_w * (-cth * bx - sth * bz)
    return Q


class ReducedAnalyticalModel:
    """Reduced longitudinal thorax-abdomen model (reduced_model_spec.md)."""

    def __init__(self, params: ReducedModelParams):
        if params.m_a == 0.0 and not params.lock_abdomen:
            raise ValueError("m_a = 0 requires lock_abdomen=True (M_singular)")
        self.p = params
        self.last_tau_act_required: float = 0.0  # reaction torque, prescribed mode

    def rhs(
        self,
        state: np.ndarray,
        t: float,
        forces: list[PointForce] | None = None,
        tau_act: float = 0.0,
        alpha_profile: AbdomenProfile | None = None,
        alpha_limits: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """xdot = [qd, qdd] for the 8-state; see module docstring for modes."""
        p = self.p
        q = np.asarray(state[:4], dtype=float)
        qd = np.asarray(state[4:], dtype=float)
        forces = forces or []

        if alpha_profile is not None:
            a, ad, add = prescribed_alpha(alpha_profile, t, alpha_limits)
            q[3], qd[3] = a, ad
        M = mass_matrix(q, p)
        bias = bias_vector(q, qd, p)
        G = gravity_vector(q, p)
        Q = generalized_force(q[2], forces)
        Q += np.array([-p.c_lin * qd[0], -p.c_lin * qd[1],
                       -p.c_ang * qd[2], -p.c_a * qd[3]])
        Q[3] += tau_act

        qdd = np.zeros(4)
        if alpha_profile is not None:
            # prescribed abdomen: solve body block, record required torque
            add_cmd = add
            rhs_b = (Q - bias - G)[:3] - M[:3, 3] * add_cmd
            qdd[:3] = np.linalg.solve(M[:3, :3], rhs_b)
            qdd[3] = add_cmd
            self.last_tau_act_required = float(
                M[3, :] @ qdd + bias[3] + G[3] - (Q[3] - tau_act)
            )
        elif p.lock_abdomen:
            qdd[:3] = np.linalg.solve(M[:3, :3], (Q - bias - G)[:3])
            qdd[3] = 0.0
        else:
            qdd = np.linalg.solve(M, Q - bias - G)
            self.last_tau_act_required = tau_act

        return np.concatenate([qd, qdd])

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy T + V (for conservation tests)."""
        p = self.p
        x_, z, th, al = state[:4]
        xd, zd, thd, ald = state[4:]
        xa_d = xd + p.d * np.sin(th) * thd + p.l_a * np.sin(th + al) * (thd + ald)
        za_d = zd + p.d * np.cos(th) * thd + p.l_a * np.cos(th + al) * (thd + ald)
        T = 0.5 * p.m_t * (xd**2 + zd**2) + 0.5 * p.I_t * thd**2 \
            + 0.5 * p.m_a * (xa_d**2 + za_d**2)
        za = z + p.d * np.sin(th) + p.l_a * np.sin(th + al)
        V = -(p.m_t * z + p.m_a * za) * p.g
        return float(T + V)


def cross_check_against_reference(n: int = 100, seed: int = 0) -> float:
    """Max relative deviation of the hand-written terms vs the SymPy reference."""
    rng = np.random.default_rng(seed)
    p = ReducedModelParams(m_t=0.011, m_a=0.004, I_t=2e-6, d=0.015, l_a=0.015)
    worst = 0.0
    for _ in range(n):
        th, al = rng.uniform(-1.0, 1.0, 2)
        thd, ald = rng.uniform(-5.0, 5.0, 2)
        q = np.array([0.0, 0.0, th, al])
        qd = np.array([0.0, 0.0, thd, ald])
        M_ref = ref.free_M(th, al, p.m_t, p.m_a, p.I_t, p.d, p.l_a)
        b_ref = ref.free_bias(th, al, thd, ald, p.m_t, p.m_a, p.d, p.l_a)
        G_ref = ref.free_G(th, al, p.m_t, p.m_a, p.I_t, p.d, p.l_a, p.g)
        for name, got, want in (("M", mass_matrix(q, p), M_ref),
                                ("bias", bias_vector(q, qd, p), b_ref),
                                ("G", gravity_vector(q, p), G_ref)):
            scale = max(float(np.abs(want).max()), 1e-12)
            worst = max(worst, float(np.abs(got - want).max()) / scale)
    return worst
