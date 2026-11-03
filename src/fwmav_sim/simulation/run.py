"""Experiment orchestration for the M1 fixed-body scope."""

from __future__ import annotations

from pathlib import Path

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.experiment.record import save_record
from fwmav_sim.simulation.fixed_body import run_fixed_body


def run_experiment(experiment_path: str | Path, out_root: str | Path = "outputs") -> Path:
    """Run a fixed-body experiment YAML and save the run record.

    Returns the record directory. Emits the validity section required by the
    stage file (assumption IDs wired into the record) and records the
    per-wing Reynolds range so regime validity is inspectable.
    """
    experiment_path = Path(experiment_path)
    exp = load_experiment(experiment_path)
    if exp.kind != "fixed_body":
        raise ValueError(f"M1 supports kind='fixed_body' only, got {exp.kind!r}")
    vehicle_path = experiment_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)

    ts = run_fixed_body(vehicle, exp.kinematics, env, model,
                        exp.n_wingbeats, exp.steps_per_wingbeat)

    re_range = {}
    for w in vehicle.wings:
        re = ts[f"{w.name}/reynolds"]
        re_range[w.name] = {"min": float(re.min()), "max": float(re.max())}

    extra_meta = {
        "experiment_file": str(experiment_path),
        "vehicle_file": str(vehicle_path),
        "reynolds_range": re_range,
        "validity": {
            "assumption_ids": ts.meta["assumptions"],
            "warnings": [
                "Quasi-steady coefficients (Dickinson et al. 1999) are Robofly-regime fits; "
                "their validity for this vehicle is unverified without experimental data.",
                "Wake capture, clap-and-fling and wing deformation are excluded (A-002, A-005).",
            ],
        },
    }
    return save_record(out_root, exp.name, ts, experiment_path, vehicle_path, extra_meta)
