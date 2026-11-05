"""Probe 2: base-motion-only bias, RNEA variants vs Jacobian reference."""
import numpy as np

from fwmav_sim.core.transforms import (
    quat_from_euler,
)
from fwmav_sim.dynamics.spatial import ArticulatedModel
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}

# geometry matching the M2 reduced-model parameters
_M_T, _M_A = 0.011, 0.004
_IT, _IA = 2.0e-6, 5.0e-7
_D, _L_A = 0.015, 0.015


def _vehicle() -> VehicleDefinition:
    return VehicleDefinition.model_validate({
        "name": "m3_spatial_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": _M_T, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[_IT, 0, 0], [0, _IT, 0], [0, 0, _IT]],
                         "unit": "kg m^2", "provenance": _PROV}},
            {"name": "abdomen_roll_carrier",
             "mass": {"value": 0.0, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[0.0] * 3] * 3, "unit": "kg m^2", "provenance": _PROV},
             "massless": True},
            {"name": "abdomen",
             "mass": {"value": _M_A, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [-_L_A, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[_IA, 0, 0], [0, _IA, 0], [0, 0, _IA]],
                         "unit": "kg m^2", "provenance": _PROV}},
        ],
        "joints": [
            {"name": "abdomen_pitch", "type": "revolute",
             "parent_body": "thorax", "child_body": "abdomen_roll_carrier",
             "axis": [0.0, 1.0, 0.0],
             "location": {"value": [-_D, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
            {"name": "abdomen_roll", "type": "revolute",
             "parent_body": "abdomen_roll_carrier", "child_body": "abdomen",
             "axis": [1.0, 0.0, 0.0],
             "location": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
        ],
        "wings": [],
        "aero": {"name": "none", "coefficients": {}},
    })



from derivations.probe_bias import jacobian_bias

model = ArticulatedModel(_vehicle())

q = np.concatenate([[0.02, 0.01, -0.03], quat_from_euler(0.2, 0.3, -0.1), [0.25, -0.15]])
# base motion only
u = np.array([0.3, -0.5, 0.2, 0.4, 0.1, -0.2, 0.0, 0.0])

bJ = jacobian_bias(q, u)

# RNEA with and without the omega x v term in A[0]
import fwmav_sim.dynamics.spatial as sp

orig = sp.ArticulatedModel.rnea


def rnea_no_wxv(self, q, u, qdd, f_ext=None, gravity=True):
    p, R_wb, qj = self._unpack_q(q)
    X_up = self._x_up(R_wb, p, qj)
    f_ext = f_ext or {}
    S = [np.concatenate([np.asarray(j.axis, float), np.zeros(3)]) for j in self.joints]
    V = [None] * (self.nj + 1)
    A = [None] * (self.nj + 1)
    V[0] = u[:6]
    A[0] = qdd[:6].copy()  # NO omega x v
    if gravity:
        A[0][3:] -= R_wb.T @ np.array([0.0, 0.0, 9.81])
    for j in range(self.nj):
        i = j + 1
        V[i] = X_up[i] @ V[j] + S[j] * u[6 + j]
        A[i] = X_up[i] @ A[j] + S[j] * qdd[6 + j] + sp.cross_motion(V[i], S[j] * u[6 + j])
    f = [None] * (self.nj + 1)
    for i in range(self.nj, -1, -1):
        name = self.body_names[i]
        f_i = self._I_body[i] @ A[i] + sp.cross_force(V[i], self._I_body[i] @ V[i])
        if name in f_ext:
            f_i = f_i - np.asarray(f_ext[name], dtype=float)
        if i < self.nj:
            f_i = f_i + X_up[i + 1].T @ f[i + 1]
        f[i] = f_i
    tau = [S[j] @ f[j + 1] for j in range(self.nj)]
    return np.concatenate([f[0], np.asarray(tau)])


bR_with = model.rnea(q, u, np.zeros(8), gravity=False)   # has omega x v in A[0]
bR_without = rnea_no_wxv(model, q, u, np.zeros(8), gravity=False)

np.set_printoptions(precision=9, suppress=True, linewidth=140)
print("Jacobian bias:  ", bJ)
print("RNEA with wxv:  ", bR_with)
print("RNEA without:   ", bR_without)
