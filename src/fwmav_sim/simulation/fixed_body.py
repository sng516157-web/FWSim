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
from fwmav_sim.dynamics.wing_inertia import (
    wing_inertial_state,
    wing_mass_properties,
)
from fwmav_sim.kinematics.blade_element import discretize, element_state
from fwmav_sim.vehicle.schema import VehicleDefinition


def run_fixed_body(
    vehicle: VehicleDefinition,
    kinematics,
    env: Environment,
    model: AerodynamicModel,
    n_wingbeats: int,
    steps_per_wingbeat: int,
) -> TimeSeries:
    """Time-stepped fixed-body force simulation over n wingbeats.

    kinematics: one PrescribedWingKinematics (broadcast) or a per-wing dict
    (all wings must share the flapping frequency; phases may differ)."""
    if isinstance(kinematics, dict):
        freqs = {round(k.flap_frequency.value, 12) for k in kinematics.values()}
        if len(freqs) != 1:
            raise ValueError("all wings must share the flapping frequency")
        f = next(iter(kinematics.values())).flap_frequency.value
    else:
        kinematics = {w.name: kinematics for w in vehicle.wings}
        f = kinematics and next(iter(kinematics.values())).flap_frequency.value
    period = 1.0 / f
    t = np.arange(n_wingbeats * steps_per_wingbeat) * (period / steps_per_wingbeat)
    n = len(t)

    wings = {w.name: w for w in vehicle.wings}
    be = vehicle.blade_elements
    elements = {
        name: discretize(w, be.n_elements, be.discretization) for name, w in wings.items()
    }
    # wing structural mass (DR-006): inertial root force/moment channels for
    # wings that carry a mass — what a force balance measures on a rig on top
    # of the aerodynamic force (F_sensor = F_aero + F_inertial)
    inertia_props = {name: wing_mass_properties(w) for name, w in wings.items()}
    unsteady = vehicle.aero.unsteady or {}
    unsteady_on = bool(unsteady)

    channels: dict[str, np.ndarray] = {}
    for name, w in wings.items():
        m = len(elements[name])
        for key, shape in {
            "F_trans": (n, 3), "F_rot": (n, 3), "F_am": (n, 3), "F_total": (n, 3),
            "M_total": (n, 3), "P_aero": (n,), "element_force": (n, m, 3),
            "alpha": (n, m), "reynolds": (n, m),
        }.items():
            channels[f"{name}/{key}"] = np.zeros(shape)
        if inertia_props[name] is not None:
            for key in ("F_inertial", "M_inertial", "F_sensor"):
                channels[f"{name}/{key}"] = np.zeros((n, 3))
        if unsteady_on:
            channels[f"{name}/F_unsteady"] = np.zeros((n, 3))

    # mirror-pair tip positions for the clap-and-fling term (M6): pair each
    # wing with the first wing of the opposite side
    mirror_of: dict[str, str | None] = {}
    if "clap_fling" in unsteady:
        for name, w in wings.items():
            mirror_of[name] = next((n2 for n2, w2 in wings.items() if w2.side != w.side), None)

    from fwmav_sim.kinematics.blade_element import (
        batched_element_state,
    )

    for i, ti in enumerate(t):
        tips = {}
        if "clap_fling" in unsteady:
            from fwmav_sim.aero.unsteady import _wing_tip_W

            tips = {name: _wing_tip_W(w, kinematics[name], ti) for name, w in wings.items()}
        for name, w in wings.items():
            # batched blade-element evaluation (M6.75 perf pass)
            st = batched_element_state(w, elements[name], kinematics[w.name], ti,
                                       nu=env.nu, v_air_W=env.v_air_W)
            bf = model.element_forces_batch(st, env)
            F_el = bf["F_trans"] + bf["F_rot"] + bf["F_am"]
            if unsteady_on:
                # unsteady terms stay on the scalar path (per-element aux)
                els = elements[name]
                F_un = np.zeros(3)
                for j in range(len(els)):
                    st_j = element_state(w, els[j], kinematics[w.name], ti,
                                         nu=env.nu, v_air_W=env.v_air_W)
                    st_j.aux = {"wing": w, "kin": kinematics[w.name],
                                "mirror_tip_W": tips.get(mirror_of.get(name)),
                                "own_tip_W": tips.get(name)}
                    ef_j = model.element_forces(st_j, env)
                    if ef_j.F_unsteady is not None:
                        F_un += ef_j.F_unsteady
                        F_el[j] += ef_j.F_unsteady
                channels[f"{name}/F_unsteady"][i] = F_un
            F_sum = F_el.sum(axis=0)
            M_sum = np.cross(st["r_W"], F_el).sum(axis=0)
            P = -float((F_el * st["v_W"]).sum())
            channels[f"{name}/F_trans"][i] = bf["F_trans"].sum(axis=0)
            channels[f"{name}/F_rot"][i] = bf["F_rot"].sum(axis=0)
            channels[f"{name}/F_am"][i] = bf["F_am"].sum(axis=0)
            channels[f"{name}/F_total"][i] = F_sum
            channels[f"{name}/M_total"][i] = M_sum
            channels[f"{name}/P_aero"][i] = P
            channels[f"{name}/element_force"][i] = F_el
            channels[f"{name}/alpha"][i] = st["alpha"]
            channels[f"{name}/reynolds"][i] = st["reynolds"]
            props = inertia_props[name]
            if props is not None:
                st_i = wing_inertial_state(w, kinematics[w.name], props, ti,
                                           np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
                F_i, N_i, c_W = st_i.wrench_on_thorax(np.zeros(3), np.zeros(3))
                channels[f"{name}/F_inertial"][i] = F_i
                channels[f"{name}/M_inertial"][i] = np.cross(c_W, F_i) + N_i
                channels[f"{name}/F_sensor"][i] = F_sum + F_i

    meta = {
        "kind": "fixed_body",
        "n_wingbeats": n_wingbeats,
        "steps_per_wingbeat": steps_per_wingbeat,
        "flap_frequency_hz": f,
        "wingbeat_period_s": period,
        "vehicle": vehicle.name,
        "aero_model": type(model).__name__,
        "wing_inertia": {name: props.mass for name, props in inertia_props.items()
                         if props is not None},
        "assumptions": ["A-001", "A-002", "A-003", "A-004", "A-005", "A-006"]
                       + (["A-010"] if any(p is not None for p in inertia_props.values())
                          else []),
    }
    return TimeSeries(t=t, channels=channels, meta=meta)
