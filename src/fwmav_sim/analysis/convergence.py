"""Blade-count x steps-per-wingbeat convergence study (M1 Task 12).

    python -m fwmav_sim.analysis.convergence experiments/m1_convergence.yaml

Runs the fixed-body experiment on a grid of (n_elements, steps_per_wingbeat),
computes cycle-averaged resultant force per wing, and reports the L2
difference against the finest case. The table is stored in the run record
(JSON) and printed verbatim for the milestone report. A run is never
"converged" because it reached its end time (brief §5.7).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.experiment.config import load_vehicle
from fwmav_sim.experiment.record import git_hash
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.simulation.fixed_body import run_fixed_body
from fwmav_sim.vehicle.schema import BladeElementSpec, EnvironmentSpec


class ConvergenceExperiment(BaseModel):
    """Grid definition for the M1 convergence study."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: str = "convergence"
    vehicle: str  # path relative to this file
    kinematics: PrescribedWingKinematics
    environment: EnvironmentSpec
    n_wingbeats: int = Field(gt=0, default=3)
    blade_counts: list[int]
    steps_per_wingbeat: list[int]


def run_convergence(config_path: str | Path, out_root: str | Path = "outputs") -> Path:
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as fh:
        exp = ConvergenceExperiment.model_validate(yaml.safe_load(fh))
    vehicle_path = config_path.parent / exp.vehicle
    vehicle = load_vehicle(vehicle_path)
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    period = 1.0 / exp.kinematics.flap_frequency.value

    finest_key = (max(exp.blade_counts), max(exp.steps_per_wingbeat))
    means: dict[tuple[int, int], np.ndarray] = {}
    for n_el in exp.blade_counts:
        for steps in exp.steps_per_wingbeat:
            vehicle_run = vehicle.model_copy(
                update={"blade_elements": BladeElementSpec(
                    n_elements=n_el, discretization=vehicle.blade_elements.discretization
                )}
            )
            ts = run_fixed_body(vehicle_run, exp.kinematics, env, model,
                                exp.n_wingbeats, steps)
            per_wing = []
            for w in vehicle.wings:
                # mean of the LAST cycle's mean is reported; spread across
                # cycles is small for prescribed kinematics (periodic forcing)
                mean, spread = wingbeat_mean(ts, f"{w.name}/F_total", period)
                per_wing.append(np.asarray(mean))
            means[(n_el, steps)] = np.concatenate(per_wing)
            print(f"N={n_el:3d} steps={steps:5d}  done", flush=True)

    ref = means[finest_key]
    table = []
    for (n_el, steps), m in means.items():
        l2 = float(np.linalg.norm(m - ref) / np.linalg.norm(ref))
        table.append({"n_elements": n_el, "steps_per_wingbeat": steps,
                      "rel_L2_vs_finest": l2})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"{stamp}_{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "name": exp.name,
        "git_hash": git_hash(),
        "reference": {"n_elements": finest_key[0], "steps_per_wingbeat": finest_key[1]},
        "reference_mean_forces_N": ref.tolist(),
        "table": table,
        "note": "relative L2 of concatenated per-wing cycle-averaged resultant force "
                "vs the finest case",
    }
    out_dir.joinpath("convergence.json").write_text(json.dumps(record, indent=2))

    print(f"\nreference (finest): N={finest_key[0]}, steps={finest_key[1]}")
    print(f"{'N':>4} {'steps/wb':>8} {'rel L2':>12}")
    for row in table:
        print(f"{row['n_elements']:>4} {row['steps_per_wingbeat']:>8} "
              f"{row['rel_L2_vs_finest']:>12.3e}")
    print(f"\nrecord: {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m fwmav_sim.analysis.convergence <config.yaml>")
        sys.exit(1)
    run_convergence(sys.argv[1])
