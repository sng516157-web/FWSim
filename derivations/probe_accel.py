"""Probe 4: acceleration recursion conventions.

Variant A (current RNEA): A_i = X_up A_parent + S qdd + V_i x (S qd_i)
Variant B (componentwise): A_i = X_up A_parent + dX_up/dt V_parent + S qdd
Reference: A_i = J_i u_dot + J_dot_i u  (componentwise, from body_jacobians FD)
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "derivations")
from probe_bias import _vehicle  # noqa: E402

from fwmav_sim.core.transforms import Rx, Ry, quat_from_euler  # noqa: E402
from fwmav_sim.dynamics.spatial import ArticulatedModel, cross_motion  # noqa: E402

model = ArticulatedModel(_vehicle())
NJ = model.nj
q = np.concatenate([[0.0, 0.0, 0.0], quat_from_euler(0.0, 0.3, 0.0), [0.2, 0.1]])
u = np.array([0.3, -0.5, 0.2, 0.4, 0.1, -0.2, 0.8, 0.5])
udot = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, 0.6, -0.3])

_, R_wb, qj = model._unpack_q(q)
X_up = model._x_up(R_wb, np.zeros(3), qj)

# --- reference: A_i = J_i udot + Jdot_i u  (FD of jacobians along qdot)
from probe_bias import u_of_q_qdot  # noqa: E402


def quat_normalize_safe(v):
    return v / np.linalg.norm(v)


qd = u_of_q_qdot(q, u)
eps = 1e-7
Jd = []
for i in range(NJ + 1):
    Jdi = np.zeros((6, 8))
    for k in range(7 + NJ):
        dq = np.zeros(7 + NJ)
        dq[k] = eps
        qp = q + dq
        qp[3:7] = quat_normalize_safe(qp[3:7])
        qm = q - dq
        qm[3:7] = quat_normalize_safe(qm[3:7])
        Jdi += (model.body_jacobians(qp)[i] - model.body_jacobians(qm)[i]) / (2 * eps) * qd[k]
    Jd.append(Jdi)



Js = model.body_jacobians(q)
A_ref = [Js[i] @ udot + Jd[i] @ u for i in range(NJ + 1)]

# --- variant A: current RNEA recursion
def accel_variant_A():
    S = [np.concatenate([np.asarray(j.axis, float), np.zeros(3)]) for j in model.joints]
    V = [None] * (NJ + 1)
    A = [None] * (NJ + 1)
    V[0] = u[:6]; A[0] = udot[:6]
    for j in range(NJ):
        i = j + 1
        V[i] = X_up[i] @ V[j] + S[j] * u[6 + j]
        A[i] = X_up[i] @ A[j] + S[j] * udot[6 + j] + cross_motion(V[i], S[j] * u[6 + j])
    return A

# --- variant B: componentwise with dX_up/dt
def accel_variant_B():
    S = [np.concatenate([np.asarray(j.axis, float), np.zeros(3)]) for j in model.joints]
    V = [None] * (NJ + 1)
    A = [None] * (NJ + 1)
    V[0] = u[:6]; A[0] = udot[:6]
    for j in range(NJ):
        i = j + 1
        joint = model.joints[j]
        axis = np.asarray(joint.axis, float)
        key = "xyz"[int(np.argmax(np.abs(axis)))]
        sign = float(np.sign(axis[np.argmax(np.abs(axis))]))
        rot = {"x": Rx, "y": Ry}[key]
        R_jm = rot(-sign * qj[j])
        dR = (-sign) * R_jm @ np.array([[0, -axis[2], axis[1]],
                                        [axis[2], 0, -axis[0]],
                                        [-axis[1], axis[0], 0.0]])
        r = np.asarray(joint.location.value, float)
        # dX/dt = [[dR, 0], [-dR skew(r), dR]] * qd_j
        dX = np.zeros((6, 6))
        dX[:3, :3] = dR
        dX[3:, :3] = -dR @ np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0.0]])
        dX[3:, 3:] = dR
        V[i] = X_up[i] @ V[j] + S[j] * u[6 + j]
        A[i] = X_up[i] @ A[j] + (dX * u[6 + j]) @ V[j] + S[j] * udot[6 + j]
    return A

np.set_printoptions(precision=8, suppress=True, linewidth=150)
for i, name in enumerate(model.body_names):
    Aa = accel_variant_A()[i]
    Ab = accel_variant_B()[i]
    print(f"{name:22s} |ref-A|={np.abs(A_ref[i]-Aa).max():.3e}  |ref-B|={np.abs(A_ref[i]-Ab).max():.3e}")
