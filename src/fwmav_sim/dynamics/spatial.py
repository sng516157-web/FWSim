"""Spatial-vector articulated dynamics (M3 Task 2; DR-001).

Featherstone-style spatial algebra in NumPy, restricted to the MVP topology:
one 6-DoF floating base (thorax) + a small serial revolute chain (abdomen
pitch, abdomen roll). n = 8 generalized coordinates; M(q) is factorized
directly (no ABA needed, DR-001 §1.2).

Conventions (self-contained; verified against 4x4 homogeneous transforms and
the M2 SymPy reference in tests):
- Spatial motion vector V = [ω; v], spatial force F = [n; f].
- DCM convention unchanged: R_A_B maps B-coords -> A-coords (frames.md).
- Motion transform xform_motion(R_A_B, p) maps V_B -> V_A where p is B's
  origin in A coords: Ad_[R,p] = [[R, 0], [skew(p) R, R]].
- Force transform: F_A = Ad^{-T} F_B = [[R, skew(p) R], [0, R]].
- Spatial inertia about the body-frame origin o, CoM at c (body coords):
  I_o = [[I_c + m skew(c)' skew(c), m skew(c)], [m skew(c)', m I_3]].

Model state: q_model = (p_B[3], quat[4] scalar-first, q_joints[nj]).
Model velocity: u = [ω_B(3); v_B(3); q̇_joints(nj)] — angular first, in the
base (thorax) frame (Featherstone ordering). Adapters convert to/from the
integrator layout in simulation/integrator.py.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import Rx, Ry, Rz, quat_to_matrix, skew


def xform_motion(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Spatial motion transform V_A = X V_B; p = origin of B in A coords."""
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, :3] = skew(p) @ R
    X[3:, 3:] = R
    return X


def xform_force(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Spatial force transform F_A = X* F_B (dual of xform_motion)."""
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[:3, 3:] = skew(p) @ R
    X[3:, 3:] = R
    return X


def cross_motion(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Spatial cross product [ω;v] x [mω;mv] = [ωxmω + vxmv; ωxmv]."""
    w, vv = v[:3], v[3:]
    mw, mv = m[:3], m[3:]
    return np.concatenate([np.cross(w, mw) + np.cross(vv, mv), np.cross(w, mv)])


def cross_force(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Spatial cross-force [ω;v] x* [n;f] = [ωxn + vxf; ωxf]."""
    w, vv = v[:3], v[3:]
    n, ff = f[:3], f[3:]
    return np.concatenate([np.cross(w, n) + np.cross(vv, ff), np.cross(w, ff)])


def spatial_inertia(mass: float, com: np.ndarray, I_c: np.ndarray) -> np.ndarray:
    """Spatial inertia about the frame origin; com = CoM offset in that frame,
    I_c = inertia about the CoM."""
    cx = skew(com)
    I_sp = np.zeros((6, 6))
    I_sp[:3, :3] = I_c + mass * cx.T @ cx
    I_sp[:3, 3:] = mass * cx
    I_sp[3:, :3] = mass * cx.T
    I_sp[3:, 3:] = mass * np.eye(3)
    return I_sp


_AXIS_ROT = {"x": Rx, "y": Ry, "z": Rz}


class ArticulatedModel:
    """Floating base + serial revolute chain, built from a VehicleDefinition.

    Chain: base (root body) -> joint[0] -> body[1] -> joint[1] -> body[2] ...
    Child body frame origin = joint location in the parent frame; the joint
    frame is parent-aligned; the child frame is rotated by the joint angle
    about the joint axis.
    """

    def __init__(self, vehicle):
        bodies = {b.name: b for b in vehicle.bodies}
        parents = {j.child_body: j for j in vehicle.joints}
        self.root_name = next(b.name for b in vehicle.bodies if b.name not in parents)
        # chain order from the root
        self.joints = []
        self.body_names = [self.root_name]
        child_of = {j.parent_body: j for j in vehicle.joints}
        current = self.root_name
        while current in child_of:
            joint = child_of[current]
            self.joints.append(joint)
            self.body_names.append(joint.child_body)
            current = joint.child_body
        self.nj = len(self.joints)
        self.nv = 6 + self.nj
        self._I_body = []
        for name in self.body_names:
            b = bodies[name]
            self._I_body.append(
                spatial_inertia(b.mass.value, np.asarray(b.com_offset.value, float),
                                np.asarray(b.inertia.value, float))
            )
        self._mass = {name: bodies[name].mass.value for name in self.body_names}
        self._com_off = {name: np.asarray(bodies[name].com_offset.value, float)
                         for name in self.body_names}

    # --- configuration handling ----------------------------------------------

    def _unpack_q(self, q: np.ndarray):
        p = np.asarray(q[:3], dtype=float)
        R_wb = quat_to_matrix(np.asarray(q[3:7], dtype=float))
        qj = np.asarray(q[7:7 + self.nj], dtype=float)
        return p, R_wb, qj

    def _x_up(self, R_wb: np.ndarray, p: np.ndarray, qj: np.ndarray) -> list[np.ndarray]:
        """Per-body motion transforms child -> parent (index 0 = world->base)."""
        out = [xform_motion(R_wb, p)]  # base: world -> B
        for j, joint in enumerate(self.joints):
            axis = np.asarray(joint.axis, dtype=float)
            key = "xyz"[int(np.argmax(np.abs(axis)))]
            if not np.isclose(abs(axis).sum(), 1.0):
                raise NotImplementedError("coordinate-axis joints only")
            sign = float(np.sign(axis[np.argmax(np.abs(axis))]))
            R_j = _AXIS_ROT[key](sign * qj[j])  # child -> parent rotation about the axis
            out.append(xform_motion(R_j, np.asarray(joint.location.value, float)))
        return out

    # --- kinematics -------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """(R_W_body, origin_W) per body."""
        p, R_wb, qj = self._unpack_q(q)
        out = {self.root_name: (R_wb, p)}
        R_parent, o_parent = R_wb, p
        for j, joint in enumerate(self.joints):
            axis = np.asarray(joint.axis, dtype=float)
            key = "xyz"[int(np.argmax(np.abs(axis)))]
            sign = float(np.sign(axis[np.argmax(np.abs(axis))]))
            R_j = _AXIS_ROT[key](sign * qj[j])
            R_c = R_parent @ R_j
            o_c = o_parent + R_parent @ np.asarray(joint.location.value, float)
            out[joint.child_body] = (R_c, o_c)
            R_parent, o_parent = R_c, o_c
        return out

    def com(self, q: np.ndarray) -> np.ndarray:
        """Whole-vehicle CoM in world (brief §6.2) from the FK geometry."""
        tf = self.forward_kinematics(q)
        total = sum(self._mass.values())
        return sum(
            self._mass[name] * (tf[name][1] + tf[name][0] @ self._com_off[name])
            for name in self.body_names
        ) / total

    def body_jacobians(self, q: np.ndarray) -> list[np.ndarray]:
        """Spatial Jacobians J_i (6 x n) mapping u -> V_i (body-i coords)."""
        _, R_wb, _ = self._unpack_q(q)
        qj = np.asarray(q[7:7 + self.nj])
        # child -> parent motion transforms for the joints (current config)
        Xj = []
        S = []
        for j, joint in enumerate(self.joints):
            axis = np.asarray(joint.axis, dtype=float)
            key = "xyz"[int(np.argmax(np.abs(axis)))]
            sign = float(np.sign(axis[np.argmax(np.abs(axis))]))
            R_j = _AXIS_ROT[key](sign * qj[j])
            Xj.append(xform_motion(R_j, np.asarray(joint.location.value, float)))
            S.append(np.concatenate([axis, np.zeros(3)]))
        Js = [np.hstack([np.eye(6), np.zeros((6, self.nj))])]  # base: identity
        J_prev = Js[0]
        for j in range(self.nj):
            J_new = np.zeros((6, self.nv))
            J_new[:, :6] = Xj[j] @ J_prev[:, :6]
            for k in range(j):
                J_new[:, 6 + k] = Xj[j] @ J_prev[:, 6 + k]
            J_new[:, 6 + j] = S[j]
            Js.append(J_new)
            J_prev = J_new
        return Js

    # --- dynamics ---------------------------------------------------------------

    def crba(self, q: np.ndarray) -> np.ndarray:
        """M(q) via composite-rigid-body assembly from the body Jacobians
        (identical content to Featherstone CRBA; n = 8, DR-001)."""
        Js = self.body_jacobians(q)
        M = np.zeros((self.nv, self.nv))
        for J, I_b in zip(Js, self._I_body, strict=True):
            M += J.T @ I_b @ J
        return M

    def rnea(self, q: np.ndarray, u: np.ndarray, qdd: np.ndarray,
             f_ext: dict[str, np.ndarray] | None = None,
             gravity: bool = True) -> np.ndarray:
        """Recursive Newton-Euler: returns generalized forces [base wrench (6,
        in base coords); joint torques] for the given (q, u, qdd).

        bias(q, u) = rnea(q, u, qdd=0) = C(q, u) u + G(q).
        Gravity enters as a fictitious base acceleration (Featherstone trick).
        """
        p, R_wb, qj = self._unpack_q(q)
        X_up = self._x_up(R_wb, p, qj)
        f_ext = f_ext or {}

        S = [np.concatenate([np.asarray(joint.axis, float), np.zeros(3)])
             for joint in self.joints]
        V = [None] * (self.nj + 1)
        A = [None] * (self.nj + 1)
        V[0] = u[:6]
        A[0] = qdd[:6].copy()
        if gravity:
            A[0][3:] -= R_wb.T @ np.array([0.0, 0.0, 9.81])  # oppose gravity
        for j in range(self.nj):
            i = j + 1
            V[i] = X_up[i] @ V[j] + S[j] * u[6 + j]
            A[i] = X_up[i] @ A[j] + S[j] * qdd[6 + j] + cross_motion(V[i], S[j] * u[6 + j])

        # inward pass (single serial chain): f[i] = own wrench + child pulled up
        f = [None] * (self.nj + 1)
        for i in range(self.nj, -1, -1):
            name = self.body_names[i]
            f_i = self._I_body[i] @ A[i] + cross_force(V[i], self._I_body[i] @ V[i])
            if name in f_ext:
                f_i = f_i - np.asarray(f_ext[name], dtype=float)
            if i < self.nj:
                f_i = f_i + xform_force_from_motion(X_up[i + 1]) @ f[i + 1]
            f[i] = f_i
        tau = [S[j] @ f[j + 1] for j in range(self.nj)]
        return np.concatenate([f[0], np.asarray(tau)])

    def bias(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        """C(q, u) u + G(q) via RNEA with qdd = 0."""
        return self.rnea(q, u, np.zeros(self.nv))

    def forward_dynamics(self, q: np.ndarray, u: np.ndarray, tau: np.ndarray,
                         f_ext: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Solve M qdd = tau + Q_ext - bias (direct 8x8 factorization)."""
        M = self.crba(q)
        Q_ext = np.zeros(self.nv)
        if f_ext:
            Js = self.body_jacobians(q)
            for i, name in enumerate(self.body_names):
                if name in f_ext:
                    Q_ext += Js[i].T @ np.asarray(f_ext[name], dtype=float)
        rhs = np.asarray(tau, dtype=float) + Q_ext - self.bias(q, u)
        return np.linalg.solve(M, rhs)

    def reaction_torques(self, q: np.ndarray, u: np.ndarray, qdd: np.ndarray,
                         f_ext: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Internal joint torques (about each joint axis) for the given motion
        — the torque the actuator/structure must transmit (brief §6.1 item 3)."""
        out = self.rnea(q, u, qdd, f_ext=f_ext, gravity=True)
        return out[6:]


def xform_force_from_motion(X: np.ndarray) -> np.ndarray:
    """Force transform dual to a motion transform: X* = X^{-T}."""
    return np.linalg.inv(X).T
