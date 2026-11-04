"""Rotation matrices, quaternions and frame-chain composition.

Implements the frozen conventions of docs/conventions/frames.md exactly:

- DCM direction: ``R_A_B`` maps B-coords -> A-coords (parent <- child).
- Elementary rotations Rx/Ry/Rz are right-handed (frames.md preamble).
- Attitude: ZYX Euler (yaw psi, pitch theta, roll phi_r), ``R_W_B = Rz Ry Rx``.
- Quaternion: scalar-first (w, x, y, z), Hamilton, stored normalized.
- Wing chain: ``R_SP_WR_R = Rz(-phi) Rx(psi_w) Ry(theta_w)`` (right),
  ``R_SP_WR_L = Rz(+phi) Rx(-psi_w) Ry(theta_w) Rx(pi)`` (left, frames.md §4).

Symbols per docs/conventions/notation.md.
"""

from __future__ import annotations

import numpy as np

#: Reflection across the x-z plane used for left-wing mirroring (frames.md §4).
S_MIRROR = np.diag([1.0, -1.0, 1.0])


def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x @ w == cross(v, w)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast cross product for (3,) arrays — np.cross has heavy axis-machinery
    overhead that dominates the aero hot loop (M3 profiling)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def dR_dt(R: np.ndarray, omega_child: np.ndarray) -> np.ndarray:
    """Time derivative of a DCM given the child's angular velocity in child coords."""
    return R @ skew(omega_child)


# --- attitude ---------------------------------------------------------------

def R_W_B(psi: float, theta: float, phi_r: float) -> np.ndarray:
    """Body-to-world DCM from ZYX Euler angles (yaw, pitch, roll)."""
    return Rz(psi) @ Ry(theta) @ Rx(phi_r)


def R_B_SP(beta: float) -> np.ndarray:
    """Stroke-plane DCM: rotation about +y_B; positive beta raises the front (frames.md §1.4)."""
    return Ry(beta)


def R_B_SP_wing(beta: float, gamma: float = 0.0, side: str = "right") -> np.ndarray:
    """Stroke-plane DCM with optional per-wing dihedral (DR-005, frames.md §1.4):

        R_B_SP_wing = Rx(sigma * gamma) @ Ry(beta),
        sigma = -1 (right) / +1 (left), so gamma > 0 raises both wingtips.

    gamma = 0 recovers the frozen chain exactly.
    """
    if gamma == 0.0:
        return Ry(beta)
    sigma = -1.0 if side == "right" else 1.0
    return Rx(sigma * gamma) @ Ry(beta)


def R_SP_WR(phi: float, theta_w: float, psi_w: float, side: str = "right") -> np.ndarray:
    """Wing-root DCM from the wing angles (frames.md §3, §4).

    side: "right" -> Rz(-phi) Rx(psi_w) Ry(theta_w);
          "left"  -> Rz(+phi) Rx(-psi_w) Ry(theta_w) Rx(pi).
    """
    if side == "right":
        return Rz(-phi) @ Rx(psi_w) @ Ry(theta_w)
    if side == "left":
        return Rz(phi) @ Rx(-psi_w) @ Ry(theta_w) @ Rx(np.pi)
    raise ValueError(f"side must be 'right' or 'left', got {side!r}")


def euler_from_R_W_B(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of R_W_B; returns (psi, theta, phi_r). Undefined at gimbal lock."""
    theta = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    psi = np.arctan2(R[1, 0], R[0, 0])
    phi_r = np.arctan2(R[2, 1], R[2, 2])
    return psi, theta, phi_r


# --- quaternions (scalar-first, Hamilton) ------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("cannot normalize the zero quaternion")
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, scalar-first (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """DCM equivalent of q (maps body coords -> parent coords), frames.md §1.3."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion from a proper rotation matrix (branch algorithm)."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    if q[0] < 0.0:
        q = -q  # canonical hemisphere: scalar part non-negative
    return quat_normalize(q)


def quat_from_euler(psi: float, theta: float, phi_r: float) -> np.ndarray:
    """Quaternion for R_W_B(psi, theta, phi_r) via quaternion ZYX composition."""
    def _axis_quat(angle: float, axis: int) -> np.ndarray:
        q = np.zeros(4)
        q[0] = np.cos(angle / 2.0)
        q[axis] = np.sin(angle / 2.0)
        return q

    return quat_multiply(
        quat_multiply(_axis_quat(psi, 3), _axis_quat(theta, 2)), _axis_quat(phi_r, 1)
    )


def quat_rate(q: np.ndarray, omega_B: np.ndarray) -> np.ndarray:
    """Quaternion rate from body angular velocity: q̇ = ½ q ⊗ [0, ω_B] (frames.md §1.3)."""
    return 0.5 * quat_multiply(q, np.concatenate([[0.0], omega_B]))


def quat_propagate_constant_omega(q0: np.ndarray, omega_B: np.ndarray, t: float) -> np.ndarray:
    """Exact propagation under constant body angular velocity (used in tests)."""
    q0 = quat_normalize(q0)
    w = np.linalg.norm(omega_B)
    if w == 0.0:
        return q0
    delta = np.concatenate([[np.cos(w * t / 2.0)], omega_B / w * np.sin(w * t / 2.0)])
    return quat_normalize(quat_multiply(q0, delta))
