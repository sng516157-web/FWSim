"""Instantaneous whole-vehicle mass properties (M2 Task 2).

Implements brief §6.2 verbatim from the actual body geometry and
configuration — never from a visualization marker:

    r_CoM(q) = sum_i m_i r_i(q) / sum_i m_i
    I_CoM(q) = sum_i [ R_i I_i R_i^T + m_i ( ||d_i||^2 I - d_i d_i^T ) ],
    d_i = r_i - r_CoM

Configuration layout (M2; the M3 spatial model will reuse the same body
geometry): q = [x, y, z, psi, theta, phi_r, alpha_abd] — thorax pose
(position of the B origin, ZYX Euler attitude) plus the abdomen pitch angle
alpha_abd (positive = tip down, frozen in reduced_model_spec.md §2).

Body offsets: BodySpec.com_offset is the CoM position in the *parent* frame at
zero joint configuration; a revolute joint rotates the child about the joint
axis at the joint location (JointSpec.location, axis in parent frame).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import R_W_B, Ry

G = 9.81  # gravity magnitude, m/s^2; world vector is [0, 0, +G] (NED, frames.md §1.1)

#: layout indices of the M2 configuration vector
Q_POS = slice(0, 3)
Q_EULER = slice(3, 6)  # psi, theta, phi_r
Q_ALPHA = 6


def body_com_positions(q: np.ndarray, vehicle) -> dict[str, np.ndarray]:
    """CoM position of every body in the world frame at configuration q."""
    q = np.asarray(q, dtype=float)
    p_B = q[Q_POS]
    R_wb = R_W_B(*q[Q_EULER])

    parents = {j.child_body: j for j in vehicle.joints}
    root = next(b.name for b in vehicle.bodies if b.name not in parents)

    out: dict[str, np.ndarray] = {}
    for body in vehicle.bodies:
        if body.name == root:
            out[body.name] = p_B + R_wb @ np.asarray(body.com_offset.value)
        else:
            # single-chain: child of the root via one revolute joint (M2 scope)
            j = parents[body.name]
            r_J = np.asarray(j.location.value)
            r_com0 = np.asarray(body.com_offset.value)
            if not np.allclose(j.axis, [0, 1, 0]):
                raise NotImplementedError("M2 supports revolute joints about +y_B only")
            alpha = q[Q_ALPHA]
            out[body.name] = p_B + R_wb @ (r_J + Ry(alpha) @ (r_com0 - r_J))
    return out


def body_orientations(q: np.ndarray, vehicle) -> dict[str, np.ndarray]:
    """World-frame orientation R_W_body of every body at configuration q."""
    q = np.asarray(q, dtype=float)
    R_wb = R_W_B(*q[Q_EULER])
    parents = {j.child_body: j for j in vehicle.joints}
    root = next(b.name for b in vehicle.bodies if b.name not in parents)
    out: dict[str, np.ndarray] = {}
    for body in vehicle.bodies:
        out[body.name] = R_wb if body.name == root else R_wb @ Ry(q[Q_ALPHA])
    return out


def com(q: np.ndarray, vehicle) -> np.ndarray:
    """Whole-vehicle centre of mass in the world frame (brief §6.2)."""
    positions = body_com_positions(q, vehicle)
    total = sum(b.mass.value for b in vehicle.bodies)
    return sum(b.mass.value * positions[b.name] for b in vehicle.bodies) / total


def inertia_about_com(q: np.ndarray, vehicle) -> np.ndarray:
    """Whole-vehicle inertia tensor about the instantaneous CoM (brief §6.2)."""
    positions = body_com_positions(q, vehicle)
    orientations = body_orientations(q, vehicle)
    r_com = com(q, vehicle)
    I = np.zeros((3, 3))
    for body in vehicle.bodies:
        d = positions[body.name] - r_com
        R_i = orientations[body.name]
        I_i = np.asarray(body.inertia.value, dtype=float)
        I += R_i @ I_i @ R_i.T + body.mass.value * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return I


def gravity_moment_about_com(q: np.ndarray, vehicle) -> np.ndarray:
    """Moment of gravity about the instantaneous CoM — identically zero in
    free body (brief §6.3). Exists so tests pin this to machine precision."""
    positions = body_com_positions(q, vehicle)
    r_com = com(q, vehicle)
    g_vec = np.array([0.0, 0.0, G])
    moment = np.zeros(3)
    for body in vehicle.bodies:
        moment += np.cross(positions[body.name] - r_com, body.mass.value * g_vec)
    return moment
