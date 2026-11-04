"""Manifold-aware integration for the articulated model (M3 Task 3).

State vector y = [p(3), quat(4, scalar-first normalized), q_j(nj),
                  v_W(3), omega_B(3), qd_j(nj)]

with model quasi-velocity u = [omega_B; v_B; qd_j] (Featherstone ordering in
dynamics/spatial.py). Kinematic map:
    p_dot    = v_W                       (world-frame linear velocity)
    quat_dot = 1/2 quat x [0, omega_B]   (frames.md §1.3)
    q_j_dot  = qd_j
    v_W_dot  = R_W_B (v_B_dot + omega_B x v_B)   (body->world transport)
    u_dot    = forward_dynamics(q, u, ...)

Integration uses scipy's DOP853 stepper directly (not solve_ivp) so the
quaternion is renormalized after every *accepted* step (brief §5.7) and the
run hard-stops on non-finite state or excessive motion (action plan §3.5:
never plausible-looking partial output).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import DOP853

from fwmav_sim.core.transforms import quat_normalize, quat_rate, quat_to_matrix


class SolverFailure(RuntimeError):
    """Raised on non-finite state or excessive motion; carries the failing state."""

    def __init__(self, message: str, t: float, y: np.ndarray):
        super().__init__(message)
        self.t = t
        self.y = y.copy()


@dataclass(frozen=True)
class GuardSpec:
    max_displacement: float = 100.0  # m; |p| beyond this => run is non-physical
    max_quat_norm_dev: float = 1e-6  # post-step check before renormalization


def pack_state(p, quat, qj, v_W, w_B, qd_j) -> np.ndarray:
    return np.concatenate([np.atleast_1d(p), np.atleast_1d(quat), np.atleast_1d(qj),
                           np.atleast_1d(v_W), np.atleast_1d(w_B), np.atleast_1d(qd_j)])


def unpack_state(y: np.ndarray, nj: int):
    p = y[0:3]
    quat = y[3:7]
    qj = y[7:7 + nj]
    v_W = y[7 + nj:10 + nj]
    w_B = y[10 + nj:13 + nj]
    qd_j = y[13 + nj:13 + 2 * nj]
    return p, quat, qj, v_W, w_B, qd_j


def make_rhs(model, force_fn=None, gravity: bool = True):
    """Build y_dot = f(t, y) for the articulated model.

    force_fn(t, qm, u) -> (tau (nv,), f_ext dict) or None.
    qm = [p, quat, q_j]; u = [omega_B; v_B; qd_j].
    """
    nj = model.nj

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        p, quat, qj, v_W, w_B, qd_j = unpack_state(y, nj)
        quat = quat_normalize(quat)
        R_wb = quat_to_matrix(quat)
        v_B = R_wb.T @ v_W
        qm = np.concatenate([p, quat, qj])
        u = np.concatenate([w_B, v_B, qd_j])
        tau = np.zeros(model.nv)
        f_ext = None
        if force_fn is not None:
            tau, f_ext = force_fn(t, qm, u)
        ud = model.forward_dynamics(qm, u, tau, f_ext=f_ext, gravity=gravity)
        vd_W = R_wb @ (ud[3:6] + np.cross(w_B, v_B))
        return np.concatenate([v_W, quat_rate(quat, w_B), qd_j, vd_W, ud[:3], ud[6:]])

    return rhs


def integrate_stepped(
    rhs,
    y0: np.ndarray,
    t_end: float,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_step: float = np.inf,
    guard: GuardSpec = GuardSpec(),
    sample_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Step DOP853 manually: renormalize the quaternion after every accepted
    step; hard-stop with SolverFailure on non-finite state or |p| overflow.

    Returns (t, Y) sampled at solver steps (or every sample_dt if given).
    """
    solver = DOP853(rhs, 0.0, np.asarray(y0, float), t_end,
                    rtol=rtol, atol=atol, max_step=max_step)
    ts = [0.0]
    Ys = [np.asarray(y0, float).copy()]
    grid_next = sample_dt if sample_dt else None
    while solver.status == "running":
        solver.step()
        y = solver.y.copy()
        # quaternion renormalization after every accepted step (brief §5.7)
        dev = abs(1.0 - np.linalg.norm(y[3:7]))
        if dev > guard.max_quat_norm_dev:
            raise SolverFailure(f"quaternion norm deviation {dev:.2e} exceeds guard", solver.t, y)
        y[3:7] = quat_normalize(y[3:7])
        solver.y = y
        if not np.all(np.isfinite(y)):
            raise SolverFailure("non-finite state", solver.t, y)
        if np.linalg.norm(y[0:3]) > guard.max_displacement:
            raise SolverFailure(f"|p| = {np.linalg.norm(y[0:3]):.1f} m exceeds guard", solver.t, y)
        if sample_dt:
            dense = solver.dense_output()
            while grid_next is not None and grid_next <= solver.t + 1e-15:
                ts.append(grid_next)
                Ys.append(dense(grid_next))
                grid_next += sample_dt
        else:
            ts.append(solver.t)
            Ys.append(y.copy())
    if solver.status != "finished":
        raise SolverFailure(f"solver status: {solver.status}", solver.t, solver.y.copy())
    return np.asarray(ts), np.asarray(Ys).T
