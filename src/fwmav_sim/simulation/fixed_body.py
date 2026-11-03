"""Fixed-body force simulation (M1 Task 7).

The body does not move; wing kinematics are prescribed (assumption A-004), so
the runner is direct time-stepping — no ODE solve. Resolution is specified as
steps per wingbeat (brief §5.7). Forces are returned per term, per wing, and
per element; instantaneous aerodynamic power is

    P_aero = -sum_elements F_resultant . v_element

(positive = mechanical power the wing motion must supply against the
aerodynamic force).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import AerodynamicModel, Environment
from fwmav_sim.core.timeseries import TimeSeries
from fwmav_sim.kinematics.blade_element import discretize, element_state
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import VehicleDefinition


def run_fixed_body(
    vehicle: VehicleDefinition,
    kinematics: PrescribedWingKinematics,
    env: Environment,
    model: AerodynamicModel,
    n_wingbeats: int,
    steps_per_wingbeat: int,
) -> TimeSeries:
    """Time-stepped fixed-body force simulation over n wingbeats."""
    f = kinematics.flap_frequency.value
    period = 1.0 / f
    t = np.arange(n_wingbeats * steps_per_wingbeat) * (period / steps_per_wingbeat)
    n = len(t)

    wings = {w.name: w for w in vehicle.wings}
    be = vehicle.blade_elements
    elements = {
        name: discretize(w, be.n_elements, be.discretization) for name, w in wings.items()
    }

    channels: dict[str, np.ndarray] = {}
    for name, w in wings.items():
        m = len(elements[name])
        for key, shape in {
            "F_trans": (n, 3), "F_rot": (n, 3), "F_am": (n, 3), "F_total": (n, 3),
            "M_total": (n, 3), "P_aero": (n,), "element_force": (n, m, 3),
            "alpha": (n, m), "reynolds": (n, m),
        }.items():
            channels[f"{name}/{key}"] = np.zeros(shape)

    for i, ti in enumerate(t):
        for name, w in wings.items():
            F_sum = np.zeros(3)
            M_sum = np.zeros(3)
            P = 0.0
            F_el = np.zeros((len(elements[name]), 3))
            alphas = np.zeros(len(elements[name]))
            res = np.zeros(len(elements[name]))
            for j, el in enumerate(elements[name]):
                st = element_state(w, el, kinematics, ti, nu=env.nu, v_air_W=env.v_air_W)
                ef = model.element_forces(st, env)
                F_sum += ef.resultant
                M_sum += np.cross(st.r_W, ef.resultant)
                P += -float(ef.resultant @ st.v_W)
                F_el[j] = ef.resultant
                alphas[j] = st.alpha
                res[j] = st.reynolds
                for key, vec in (("F_trans", ef.F_trans), ("F_rot", ef.F_rot), ("F_am", ef.F_am)):
                    channels[f"{name}/{key}"][i] += vec
            channels[f"{name}/F_total"][i] = F_sum
            channels[f"{name}/M_total"][i] = M_sum
            channels[f"{name}/P_aero"][i] = P
            channels[f"{name}/element_force"][i] = F_el
            channels[f"{name}/alpha"][i] = alphas
            channels[f"{name}/reynolds"][i] = res

    meta = {
        "kind": "fixed_body",
        "n_wingbeats": n_wingbeats,
        "steps_per_wingbeat": steps_per_wingbeat,
        "flap_frequency_hz": f,
        "wingbeat_period_s": period,
        "vehicle": vehicle.name,
        "aero_model": type(model).__name__,
        "assumptions": ["A-001", "A-002", "A-003", "A-004", "A-005", "A-006"],
    }
    return TimeSeries(t=t, channels=channels, meta=meta)
