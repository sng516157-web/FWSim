"""Probe: bias via J'u + gyroscopic terms (FD of body_jacobians) vs RNEA."""
import numpy as np

from fwmav_sim.core.transforms import (
    quat_from_euler,
    quat_normalize,
    quat_rate,
    quat_to_matrix,
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




model = ArticulatedModel(_vehicle())
NJ = model.nj
NQ = 7 + NJ


def u_of_q_qdot(qm, u):
    """configuration rates from quasi-velocities."""
    R_wb = quat_to_matrix(qm[3:7])
    pd = R_wb @ u[3:6]
    quatd = quat_rate(qm[3:7], u[0:3])
    return np.concatenate([pd, quatd, u[6:6 + NJ]])


def jacobian_bias(qm, u, eps=1e-7):
    Js0 = model.body_jacobians(qm)
    qdot = u_of_q_qdot(qm, u)
    bias = np.zeros(8)
    for i, I_sp in enumerate(model._I_body):
        J0 = Js0[i]
        V = J0 @ u
        from fwmav_sim.dynamics.spatial import cross_force
        f_gyro = cross_force(V, I_sp @ V)
        # Jdot via FD along qdot
        Jd = np.zeros_like(J0)
        for k in range(NQ):
            dq = np.zeros(NQ)
            dq[k] = eps
            qp = qm + dq
            qp[3:7] = quat_normalize(qp[3:7])
            qmn = qm - dq
            qmn[3:7] = quat_normalize(qmn[3:7])
            Jd += (model.body_jacobians(qp)[i] - model.body_jacobians(qmn)[i]) / (2 * eps) * qdot[k]
        f_inertial = I_sp @ (Jd @ u)
        bias += J0.T @ (f_inertial + f_gyro)
    return bias


rng = np.random.default_rng(0)
q = np.concatenate([[0.02, 0.01, -0.03], quat_from_euler(0.2, 0.3, -0.1), [0.25, -0.15]])
u = rng.normal(size=8) * 0.4

bJ = jacobian_bias(q, u)
bR = model.rnea(q, u, np.zeros(8), gravity=False)
print("Jacobian bias:", bJ.round(8))
print("RNEA bias:    ", bR.round(8))
print("diff:", np.abs(bJ - bR).max())
