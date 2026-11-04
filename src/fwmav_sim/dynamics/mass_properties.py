"""Instantaneous whole-vehicle mass properties (M2 Task 2; extended in M3).

Implements brief §6.2 verbatim from the actual body geometry and
configuration — never from a visualization marker:

    r_CoM(q) = sum_i m_i r_i(q) / sum_i m_i
    I_CoM(q) = sum_i [ R_i I_i R_i^T + m_i ( ||d_i||^2 I - d_i d_i^T ) ],
    d_i = r_i - r_CoM

Configuration layout: q = [x, y, z, psi, theta, phi_r, alpha_abd, beta_abd]
— thorax pose (position of the B origin, ZYX Euler attitude) plus one angle
per revolute joint in `vehicle.joints` order (abdomen pitch alpha_abd about
+y, then abdomen roll beta_abd about the post-pitch x; signs per
reduced_model_spec.md §2 and the vehicle YAML comments).

Frame semantics (M3): a joint's child body frame origin sits at
JointSpec.location (in the parent frame); BodySpec.com_offset is the CoM in
the body frame itself. Joint rotations compose serially along the chain.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import R_W_B, Rx, Ry, Rz

G = 9.81  # gravity magnitude, m/s^2; world vector is [0, 0, +G] (NED, frames.md §1.1)

#: layout indices of the configuration vector
Q_POS = slice(0, 3)
Q_EULER = slice(3, 6)  # psi, theta, phi_r
N_BASE = 6

_AXIS_ROT = {"x": Rx, "y": Ry, "z": Rz}


def _axis_key(axis: list[float]) -> str:
    a = np.asarray(axis, dtype=float)
    for name, ref in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        if np.allclose(a, ref):
            return name
    raise NotImplementedError(f"joint axis must be a coordinate axis, got {axis}")


def body_transforms(q: np.ndarray, vehicle) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(R_W_body, origin_W) of every body frame at configuration q."""
    q = np.asarray(q, dtype=float)
    parents = {j.child_body: j for j in vehicle.joints}
    child_of = {j.parent_body: j for j in vehicle.joints}
    root = next(b.name for b in vehicle.bodies if b.name not in parents)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        root: (R_W_B(*q[Q_EULER]), q[Q_POS].copy())
    }
    # walk the chain from the root (each body has at most one child joint)
    current = root
    j_index = 0
    while current in child_of:
        joint = child_of[current]
        R_p, o_p = out[current]
        axis_rot = _AXIS_ROT[_axis_key(joint.axis)]
        angle = q[N_BASE + j_index]
        R_c = R_p @ axis_rot(angle)
        o_c = o_p + R_p @ np.asarray(joint.location.value, dtype=float)
        out[joint.child_body] = (R_c, o_c)
        current = joint.child_body
        j_index += 1
    return out


def body_com_positions(q: np.ndarray, vehicle) -> dict[str, np.ndarray]:
    """CoM position of every body in the world frame at configuration q."""
    tf = body_transforms(q, vehicle)
    return {
        b.name: tf[b.name][1] + tf[b.name][0] @ np.asarray(b.com_offset.value, dtype=float)
        for b in vehicle.bodies
    }


def body_orientations(q: np.ndarray, vehicle) -> dict[str, np.ndarray]:
    """World-frame orientation R_W_body of every body at configuration q."""
    return {name: R for name, (R, _) in body_transforms(q, vehicle).items()}


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
    I_out = np.zeros((3, 3))
    for body in vehicle.bodies:
        d = positions[body.name] - r_com
        R_i = orientations[body.name]
        I_i = np.asarray(body.inertia.value, dtype=float)
        I_out += R_i @ I_i @ R_i.T + body.mass.value * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return I_out


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
