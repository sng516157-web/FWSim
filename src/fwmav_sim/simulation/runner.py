"""Free-body runner with failure-visible records (M3 Task 7).

Behaviour contract (action plan §3.5, brief §10 M3): non-finite state,
excessive motion, or solver failure stops the run visibly; the record is
marked FAILED with the failing state dumped. No partial output is emitted on
failure — the record's status field says what happened.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.timeseries import TimeSeries
from fwmav_sim.simulation.free_body import FreeBodyPlant
from fwmav_sim.simulation.integrator import SolverFailure, integrate_stepped
from fwmav_sim.vehicle.schema import SolverSpec


def run_free_body(
    plant: FreeBodyPlant,
    solver: SolverSpec,
    y0: np.ndarray,
    sample_hz: float = 2000.0,
) -> tuple[TimeSeries, str, dict]:
    """Run the coupled free-body simulation.

    Returns (TimeSeries, status, extra). status is "OK" or "FAILED". On
    failure the TimeSeries is empty and extra carries the failing state and
    message — the record shows FAILED, never partial-looking output.
    """
    nj = plant.nj
    max_step = 1.0 / (plant.kinematics.flap_frequency.value * solver.steps_per_wingbeat)

    status = "OK"
    extra: dict = {}
    try:
        t, Y = integrate_stepped(plant.rhs, y0, solver.max_time, rtol=solver.rtol,
                                 atol=solver.atol, max_step=max_step,
                                 sample_dt=1.0 / sample_hz)
    except SolverFailure as exc:
        return (
            TimeSeries(t=np.zeros(0), channels={}, meta={"kind": "free_body", "status": "FAILED"}),
            "FAILED",
            {"failure_message": str(exc), "failure_time": float(exc.t),
             "failing_state": exc.y.tolist()},
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        # solver-internal failure (e.g. non-finite derivatives rejected by scipy)
        return (
            TimeSeries(t=np.zeros(0), channels={}, meta={"kind": "free_body", "status": "FAILED"}),
            "FAILED",
            {"failure_message": f"{type(exc).__name__}: {exc}", "failure_time": float("nan"),
             "failing_state": []},
        )

    channels = {
        "p": Y[0:3].T,
        "quat": Y[3:7].T,
        "q_joints": Y[7:7 + nj].T,
        "v_W": Y[7 + nj:10 + nj].T,
        "omega_B": Y[10 + nj:13 + nj].T,
        "qd_joints": Y[13 + nj:13 + 2 * nj].T,
    }
    # diagnostics recomputed per sample (plant.last_* are per-call)
    n = len(t)
    alpha_cmd = np.zeros(n)
    sat = np.zeros(n, dtype=bool)
    reaction = np.zeros((n, nj))
    com = np.zeros((n, 3))
    wings = [w.name for w in plant.vehicle.wings]
    F_terms = {w: np.zeros((n, 3)) for w in wings}
    lever = {w: np.zeros((n, 3)) for w in wings}
    has_wing_inertia = bool(plant.last_F_wing_inertia) or any(
        p is not None for p in plant._wing_mass.values())
    F_inertia = {w: np.zeros((n, 3)) for w in wings} if has_wing_inertia else {}
    has_abdomen_aero = plant.abdomen_aero is not None
    F_abd = np.zeros((n, 3))
    alpha_abd_aero = np.zeros(n)
    for i in range(n):
        plant.rhs(float(t[i]), Y[:, i])  # recompute diagnostics at the sample
        alpha_cmd[i] = plant.last_alpha_cmd
        sat[i] = plant.last_saturated
        reaction[i] = plant.last_reaction
        if has_abdomen_aero and plant.last_F_abdomen_aero is not None:
            F_abd[i] = plant.last_F_abdomen_aero
            alpha_abd_aero[i] = plant.last_abdomen_alpha
        qm = np.concatenate([Y[0:3, i], Y[3:7, i], Y[7:7 + nj, i]])
        com[i] = plant.model.com(qm)
        for w in wings:
            rec = plant.last_F_wings.get(w)
            if rec:
                F_terms[w][i] = (rec["F_trans"] + rec["F_rot"] + rec["F_am"]
                                 + rec.get("F_unsteady", 0.0))
                lever[w][i] = rec["r_app_W"] - com[i]
            if has_wing_inertia:
                F_inertia[w][i] = plant.last_F_wing_inertia.get(w, np.zeros(3))
    channels["alpha_commanded"] = alpha_cmd
    channels["actuator_saturated"] = sat
    channels["reaction_torques"] = reaction
    channels["com_W"] = com
    for w in wings:
        channels[f"{w}/F_aero_total"] = F_terms[w]
        channels[f"{w}/lever_arm"] = lever[w]
        if has_wing_inertia:
            channels[f"{w}/F_wing_inertial"] = F_inertia[w]
    if has_abdomen_aero:
        channels["abdomen/F_aero"] = F_abd
        channels["abdomen/alpha_aero"] = alpha_abd_aero

    wing_masses = {w.name: plant._wing_mass[w.name].mass
                   for w in plant.vehicle.wings if plant._wing_mass[w.name] is not None}
    meta = {"kind": "free_body", "status": status,
            "solver": {"integrator": solver.integrator, "rtol": solver.rtol,
                       "atol": solver.atol,
                       "steps_per_wingbeat": solver.steps_per_wingbeat},
            "abdomen_mode": plant.control.mode,
            "wing_mass_kg": wing_masses,
            "isolation": {"mass_shift_enabled": plant.isolation.mass_shift_enabled,
                          "inertial_coupling_enabled": plant.isolation.inertial_coupling_enabled,
                          "abdomen_aero_enabled": plant.isolation.abdomen_aero_enabled,
                          "wing_inertia_enabled": plant.isolation.wing_inertia_enabled},
            "assumptions": ["A-001", "A-002", "A-003", "A-004", "A-005", "A-006", "A-008", "A-009"]
                           + (["A-010"] if wing_masses and plant.isolation.wing_inertia_enabled
                              else [])}
    return TimeSeries(t=t, channels=channels, meta=meta), status, extra
