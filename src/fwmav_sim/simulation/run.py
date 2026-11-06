"""Experiment orchestration: dispatch by experiment kind.

Kinds: fixed_body (M1), suspended_pendulum and free_reduced (M2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.constraints.suspended import SuspendedPendulumParams
from fwmav_sim.dynamics.cycle_averaged import load_average_forces
from fwmav_sim.dynamics.reduced_longitudinal import ReducedModelParams
from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.experiment.config import ExperimentDefinition, load_experiment, load_vehicle
from fwmav_sim.experiment.record import save_record
from fwmav_sim.simulation.fixed_body import run_fixed_body
from fwmav_sim.simulation.free_reduced import run_free_reduced, synthetic_thrust_law_forces
from fwmav_sim.simulation.pendulum import run_suspended_pendulum


def _profile(exp: ExperimentDefinition) -> AbdomenProfile | None:
    if exp.abdomen is None or exp.abdomen.mode != "prescribed":
        return None
    if exp.abdomen.profile is None:
        raise ValueError("abdomen mode 'prescribed' requires a profile")
    p = exp.abdomen.profile
    return AbdomenProfile(kind=p.kind, amplitude=p.amplitude, offset=p.offset,
                          frequency=p.frequency, phase=p.phase,
                          t_start=p.t_start, t_end=p.t_end)


def _alpha_limits(vehicle) -> tuple[float, float] | None:
    for j in vehicle.joints:
        if j.limits is not None:
            return tuple(j.limits)
    return None


def _run_fixed_body(exp, experiment_path):
    if not (exp.vehicle and exp.kinematics and exp.environment
            and exp.n_wingbeats and exp.steps_per_wingbeat):
        raise ValueError("fixed_body requires vehicle, kinematics, environment, "
                         "n_wingbeats, steps_per_wingbeat")
    vehicle_path = experiment_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    from fwmav_sim.experiment.config import kinematics_map
    if exp.passive_pitch:
        from fwmav_sim.dynamics.wing_pitch_compliance import run_fixed_body_passive_pitch

        ts = run_fixed_body_passive_pitch(vehicle, kinematics_map(exp, vehicle), env,
                                          model, exp.n_wingbeats, exp.steps_per_wingbeat)
    else:
        ts = run_fixed_body(vehicle, kinematics_map(exp, vehicle), env, model,
                            exp.n_wingbeats, exp.steps_per_wingbeat)
    re_range = {w.name: {"min": float(ts[f"{w.name}/reynolds"].min()),
                         "max": float(ts[f"{w.name}/reynolds"].max())}
                for w in vehicle.wings}
    coef_note = (
        f"aero coefficients: registry set '{vehicle.aero.coefficient_set}' "
        f"(regime: {model.set_info['regime']})"
        if model.set_info else
        "aero coefficients: inline from the vehicle YAML (Dickinson 1999 Robofly "
        "fits unless edited); validity for this vehicle unverified without data.")
    # fidelity ladder (M7 Task 5): the aero tier is first-class metadata
    tier = "quasi_steady+unsteady_terms" if vehicle.aero.unsteady else "quasi_steady"
    if exp.passive_pitch:
        tier += "+passive_pitch"
    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path),
        "reynolds_range": re_range,
        "validity": {
            "fidelity_tier": tier,
            "assumption_ids": ts.meta["assumptions"],
            "aero_coefficients": ({"set": vehicle.aero.coefficient_set,
                                   "regime": model.set_info["regime"],
                                   "reference": model.set_info["reference"]}
                                  if model.set_info else "inline"),
            "warnings": [
                coef_note,
                ("Reduced-order unsteady terms ACTIVE (M6): "
                 + ", ".join(sorted(vehicle.aero.unsteady))
                 + " — estimated gains, not validated (A-002 relaxed)."
                 if vehicle.aero.unsteady else
                 "Wake capture, clap-and-fling and wing deformation are excluded (A-002, A-005)."),
                *(["Passive torsional pitch active (M7): prescribed pitch ignored "
                   "for sprung wings; spring constants estimated"] if exp.passive_pitch
                  else []),
            ],
        },
    }
    return ts, vehicle_path, extra_meta


def _run_suspended_pendulum(exp, experiment_path):
    if not (exp.vehicle and exp.suspension and exp.solver):
        raise ValueError("suspended_pendulum requires vehicle, suspension, solver")
    vehicle_path = experiment_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    abdomen = exp.abdomen or None
    lock = abdomen is None or abdomen.mode == "locked"
    profile = _profile(exp)
    c_a = exp.damping.c_a if exp.damping else 0.0
    params = SuspendedPendulumParams.from_vehicle(
        vehicle, tuple(exp.suspension.pivot_body_point), c_a=c_a, lock_abdomen=lock)
    init = tuple(exp.initial_state or [0.0, 0.0, 0.0, 0.0])
    ts = run_suspended_pendulum(params, exp.solver, initial_state=init,
                                alpha_profile=profile,
                                alpha_limits=_alpha_limits(vehicle) if profile else None)
    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path),
        "validity": {
            "assumption_ids": [],
            "warnings": [
                "Suspended rigid-pivot model: pendulum agreement is NOT free-flight "
                "validation (brief §6.3, §10 M2).",
            ],
        },
    }
    return ts, vehicle_path, extra_meta


def _run_free_reduced(exp, experiment_path):
    if not exp.solver:
        raise ValueError("free_reduced requires solver")
    warnings: list[str] = []
    vehicle_path = None
    if exp.reduced_params is not None:
        rp = exp.reduced_params
        params = ReducedModelParams(m_t=rp.m_t, m_a=rp.m_a, I_t=rp.I_t, d=rp.d,
                                    l_a=rp.l_a, lock_abdomen=rp.lock_abdomen)
        vehicle_name = "(explicit reduced params)"
    elif exp.vehicle is not None:
        vehicle_path = experiment_path.parent / exp.vehicle
        vehicle = load_vehicle(vehicle_path)
        vehicle_name = vehicle.name
        lock = exp.abdomen is None or exp.abdomen.mode == "locked"
        params = ReducedModelParams.from_vehicle(vehicle, lock_abdomen=lock)
    else:
        raise ValueError("free_reduced requires vehicle or reduced_params")

    if exp.damping:
        params = ReducedModelParams(**{**params.__dict__, "c_lin": exp.damping.c_lin,
                                       "c_ang": exp.damping.c_ang, "c_a": exp.damping.c_a})

    forces = []
    inj = exp.forces
    if inj and inj.source == "synthetic_thrust_law":
        if inj.k_thrust is None or inj.base_f_hz is None or inj.arm is None:
            raise ValueError("synthetic_thrust_law requires k_thrust, base_f_hz, arm")
        forces = synthetic_thrust_law_forces(inj.k_thrust, inj.base_f_hz,
                                             inj.freq_delta_hz, inj.arm)
        warnings.append("Synthetic thrust law T = k f (RL-sandbox foil); NOT physics.")
    elif inj and inj.source == "m1_record":
        if not inj.record_path or vehicle_path is None:
            raise ValueError("m1_record injection requires record_path and a vehicle")
        vehicle = load_vehicle(vehicle_path)
        per_wing, w = load_average_forces(experiment_path.parent / inj.record_path,
                                          vehicle_name)
        warnings.extend(w)
        for wing_name, f_avg in per_wing:
            wing = next(x for x in vehicle.wings if x.name == wing_name)
            hx, _, hz = wing.hinge_location.value
            forces.append(((hx, hz), (f_avg[0], f_avg[2])))
    profile = _profile(exp)

    init = exp.initial_state or [0.0] * 8
    ts = run_free_reduced(params, exp.solver, initial_state=init, forces=forces,
                          alpha_profile=profile)
    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path) if vehicle_path else None,
        "validity": {"assumption_ids": ["A-007"], "warnings": warnings},
    }
    return ts, vehicle_path, extra_meta


def build_free_body_plant(exp, experiment_path):
    """Construct (plant, y0) exactly as _run_free_body does — the single
    physics path for free-body runs, records, and live streaming (no second
    implementation)."""
    from fwmav_sim.core.transforms import quat_from_euler
    from fwmav_sim.simulation.free_body import FreeBodyPlant, IsolationSpec
    from fwmav_sim.simulation.integrator import pack_state

    if not (exp.vehicle and exp.kinematics and exp.environment and exp.solver):
        raise ValueError("free_body requires vehicle, kinematics, environment, solver")
    vehicle_path = experiment_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)

    isolation = IsolationSpec(**(exp.isolation.model_dump() if exp.isolation else {}))
    control = _abdomen_control(exp, vehicle)
    from fwmav_sim.experiment.config import effective_base, kinematics_map

    _mode, _w = effective_base(exp)
    base_mode, release_time = "free", None
    if exp.base is not None:
        base_mode = exp.base.mode if exp.base.mode != "pivot" else "free"
        release_time = exp.base.release_time
        if release_time is not None:
            init = exp.initial_state_free
            if init and (any(init.velocity_W) or any(init.omega_B)):
                raise ValueError("base.release_time requires rest initial conditions")
    plant = FreeBodyPlant(vehicle, kinematics_map(exp, vehicle), env, model,
                          control=control, isolation=isolation,
                          base_mode=base_mode, release_time=release_time)

    init = exp.initial_state_free
    joint_angles = (init.joint_angles if init and init.joint_angles
                    else [0.0] * plant.nj)
    joint_rates = init.joint_rates if init and init.joint_rates else [0.0] * plant.nj
    y0_extra = [0.0, 0.0] if plant.n_extra else []
    y0 = np.concatenate([
        pack_state(init.position if init else [0.0] * 3,
                   quat_from_euler(*(init.euler if init else [0.0] * 3)),
                   joint_angles,
                   init.velocity_W if init else [0.0] * 3,
                   init.omega_B if init else [0.0] * 3,
                   joint_rates),
        y0_extra,
    ])
    return plant, y0, vehicle_path


def _run_free_body(exp, experiment_path):
    from fwmav_sim.simulation.runner import run_free_body

    plant, y0, vehicle_path = build_free_body_plant(exp, experiment_path)
    ts, status, extra = run_free_body(plant, exp.solver, y0)
    wing_masses = ts.meta.get("wing_mass_kg", {})
    wing_note = (
        "Wing structural mass active (DR-006, A-010): planform-derived thin-plate "
        "wing inertia folded into the base mass matrix."
        if wing_masses and plant.isolation.wing_inertia_enabled else
        "Wings massless; wing flapping actuator reaction torques not modelled (A-004)."
    )
    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path),
        "status": status,
        "schema_version": 1,
        **extra,
        "validity": {
            "assumption_ids": ts.meta.get("assumptions", []),
            "warnings": [
                "Time-resolved quasi-steady aero coupled to free-body dynamics; "
                "coefficients unverified for this vehicle (A-003, A-008).",
                "Added-mass term excludes body accelerations (A-009).",
                wing_note,
            ],
        },
    }
    return ts, vehicle_path, extra_meta


def _abdomen_control(exp, vehicle):
    from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile
    from fwmav_sim.actuation.profile_limited import ProfileLimiter
    from fwmav_sim.actuation.torque_driven import TorqueActuator
    from fwmav_sim.simulation.free_body import AbdomenControl

    abdomen = exp.abdomen
    if abdomen is None or abdomen.mode == "locked":
        return AbdomenControl(mode="locked")
    limits = _alpha_limits(vehicle)
    profile = None
    if abdomen.profile is not None:
        p = abdomen.profile
        profile = AbdomenProfile(kind=p.kind, amplitude=p.amplitude, offset=p.offset,
                                 frequency=p.frequency, phase=p.phase,
                                 t_start=p.t_start, t_end=p.t_end)
    if abdomen.mode == "prescribed":
        return AbdomenControl(mode="prescribed", profile=profile, limits=limits)
    if abdomen.mode == "profile_limited":
        if abdomen.rate_max is None or abdomen.accel_max is None:
            raise ValueError("profile_limited requires rate_max and accel_max")
        return AbdomenControl(
            mode="profile_limited", profile=profile,
            limiter=ProfileLimiter(rate_max=abdomen.rate_max, accel_max=abdomen.accel_max),
            limits=limits)
    if abdomen.mode == "torque_driven":
        if abdomen.tau_max is None:
            raise ValueError("torque_driven requires tau_max")
        amp, freq = abdomen.tau_amplitude, abdomen.tau_frequency
        return AbdomenControl(
            mode="torque_driven",
            actuator=TorqueActuator(
                tau_max=abdomen.tau_max, c_a=abdomen.joint_damping, limits=limits,
                tau_cmd=lambda t: amp * np.sin(2 * np.pi * freq * t)),
            limits=limits, joint_damping=abdomen.joint_damping)
    if abdomen.mode == "free":
        return AbdomenControl(mode="torque_driven",
                              actuator=TorqueActuator(tau_max=1e9, c_a=abdomen.joint_damping,
                                                      limits=limits),
                              limits=limits, joint_damping=abdomen.joint_damping)
    raise ValueError(f"unknown abdomen mode {abdomen.mode!r}")


def _run_closed_loop(exp, experiment_path):
    """Closed-loop hover demo (M6 Task 8): trim -> linearize -> LQR -> simulate."""
    from fwmav_sim.control.closed_loop import run_closed_loop
    from fwmav_sim.control.trim import solve_trim

    if not (exp.vehicle and exp.closed_loop and exp.solver):
        raise ValueError("closed_loop_reduced requires vehicle, closed_loop, solver")
    vehicle_path = experiment_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    params = ReducedModelParams.from_vehicle(vehicle, lock_abdomen=False)
    per_wing, w = load_average_forces(experiment_path.parent / exp.closed_loop.forces_record,
                                      vehicle.name)
    F_avg = np.zeros(3)
    x_Ts = []
    for wing_name, f_avg in per_wing:
        wing = next(x for x in vehicle.wings if x.name == wing_name)
        F_avg += np.asarray(f_avg)
        x_Ts.append(wing.hinge_location.value[0])
    r_T = (float(np.mean(x_Ts)) if x_Ts else 0.0, 0.0)
    trim = solve_trim((float(F_avg[0]), float(F_avg[2])), r_T, params)
    # cycle-averaged aerodynamic body damping (flapping counter-torque +
    # heave damping; dynamics/aero_damping.py) — required for a stabilizable
    # hover demo (M6 report §3); needs kinematics + environment in the YAML
    damping_D = None
    damp_note = "no aerodynamic damping (kinematics/environment absent): demo may diverge"
    if exp.kinematics is not None and exp.environment is not None:
        from fwmav_sim.dynamics.aero_damping import body_motion_damping, damping_matrix
        from fwmav_sim.environment.spec import environment_from_spec
        from fwmav_sim.experiment.config import kinematics_map

        d = body_motion_damping(vehicle, kinematics_map(exp, vehicle),
                                environment_from_spec(exp.environment))
        damping_D = damping_matrix(d)
        damp_note = f"aerodynamic body damping included: {d}"
    cl = exp.closed_loop
    result = run_closed_loop(
        params, trim, trim.r_T_trim,
        Q=np.diag(cl.q_diag), R=np.diag(cl.r_diag),
        x0=np.asarray(cl.initial_perturbation, float),
        t_end=exp.solver.max_time, rtol=exp.solver.rtol, atol=exp.solver.atol,
        damping_D=damping_D, du_limits=(0.5, 2e-3))
    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path),
        "status": "OK",
        "trim": {"thrust_scale": trim.thrust_scale, "alpha_abd_trim": trim.alpha_abd_trim,
                 "tau_act_trim": trim.tau_act_trim, "x_T_shift": trim.x_T_shift,
                 "theta_trim": trim.theta_trim, "r_T_trim": list(trim.r_T_trim),
                 "residual": [float(r) for r in trim.residual]},
        "lqr_gain": result.K.tolist(),
        "validity": {
            "assumption_ids": ["A-007"],
            "warnings": [
                "Design-loop demo on the Level-1 reduced model (cycle-averaged "
                "forces, A-007); NOT a flight-ready autopilot.",
                damp_note,
                *w,
            ],
        },
    }
    return result.ts, vehicle_path, extra_meta


def dispatch_experiment(experiment_path: str | Path):
    """Load and run an experiment WITHOUT saving a record.

    Returns (TimeSeries, vehicle_path, extra_meta) — the same objects
    run_experiment persists. Used by the sensitivity analysis (in-process
    reruns over patched YAMLs) and by run_experiment itself.

    Routing (DR-007): the effective base constraint may route a run to a
    different runner than the kind implies (fixed_body with base: free flies;
    free_body with base: fixed goes to the fast rig runner).
    """
    from fwmav_sim.experiment.config import effective_base

    experiment_path = Path(experiment_path)
    exp = load_experiment(experiment_path)
    mode, base_warnings = effective_base(exp)
    if exp.kind == "fixed_body":
        if mode == "free":
            if not exp.solver:
                raise ValueError("base: free on a fixed_body experiment "
                                 "requires a solver block")
            out = _run_free_body(exp, experiment_path)
        elif mode == "pivot":
            if not exp.solver:
                raise ValueError("base: pivot on a fixed_body experiment "
                                 "requires a solver block")
            out = _run_suspended_pendulum(exp, experiment_path)
        else:
            out = _run_fixed_body(exp, experiment_path)
    elif exp.kind == "suspended_pendulum":
        if mode == "free":
            out = _run_free_body(exp, experiment_path)
        elif mode == "fixed":
            out = _run_fixed_body(exp, experiment_path)
        else:
            out = _run_suspended_pendulum(exp, experiment_path)
    elif exp.kind == "free_reduced":
        out = _run_free_reduced(exp, experiment_path)
    elif exp.kind == "free_body":
        if mode == "fixed":
            if not (exp.n_wingbeats and exp.steps_per_wingbeat):
                raise ValueError("base: fixed on a free_body experiment requires "
                                 "n_wingbeats and steps_per_wingbeat (fast rig runner)")
            out = _run_fixed_body(exp, experiment_path)
        elif mode == "pivot":
            out = _run_suspended_pendulum(exp, experiment_path)
        else:
            out = _run_free_body(exp, experiment_path)
    elif exp.kind == "closed_loop_reduced":
        out = _run_closed_loop(exp, experiment_path)
    else:
        raise ValueError(f"unknown experiment kind {exp.kind!r}")
    ts, vehicle_path, meta = out
    if base_warnings:
        meta.setdefault("validity", {}).setdefault("warnings", [])
        meta["validity"]["warnings"] = base_warnings + meta["validity"]["warnings"]
    return ts, vehicle_path, meta


def run_experiment(experiment_path: str | Path, out_root: str | Path = "outputs") -> Path:
    """Run an experiment YAML and save the run record; returns the directory."""
    experiment_path = Path(experiment_path)
    exp = load_experiment(experiment_path)
    ts, vehicle_path, extra_meta = dispatch_experiment(experiment_path)
    return save_record(out_root, exp.name, ts, experiment_path, vehicle_path, extra_meta)
