"""System-identification fitting (M6 Task 9).

Fit the quasi-steady aero coefficients to a measured force time series (M4
dataset format), closing the measure -> fit -> validate loop:

    minimize over the free coefficients  sum_t |F_sim(t; c) - F_meas(t)|^2

with F_sim the fixed-body blade-element force for the dataset's kinematics.

Identifiability (documented, not hidden): the Dickinson fit form is
nonlinear in cl_freq/cl_phase/cd_freq/cd_phase; those are poorly identified
by force-only data. The default free set is therefore the SCALE parameters
{cl_0, cl_slope, cd_0, cd_slope}; the shape parameters stay at their prior
values unless ``free`` is overridden. Fitted values are written back with
source: fitted + the dataset reference (brief §5.8) — a dataset used for
fitting must be labelled calibration, never validation (M4 importer rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import REQUIRED_COEFFICIENTS, QuasiSteadyBladeElement
from fwmav_sim.kinematics.blade_element import discretize, element_state
from fwmav_sim.vehicle.schema import Parameter, Provenance

DEFAULT_FREE = ("cl_0", "cl_slope", "cd_0", "cd_slope")


@dataclass(frozen=True)
class SysidResult:
    coefficients: dict[str, float]  # full 8-parameter set (fitted + held)
    free: tuple[str, ...]
    rms_residual: float
    rms_baseline: float  # residual at the starting coefficients
    n_samples: int


def _simulate_forces(vehicle, kinematics, env, coef_vec: np.ndarray,
                     free: tuple[str, ...], c0: dict[str, float],
                     t: np.ndarray) -> np.ndarray:
    """(n, 3) summed wing force for coefficient perturbation coef_vec."""
    coefs = dict(c0)
    for k, v in zip(free, coef_vec, strict=True):
        coefs[k] = float(v)
    prov = Provenance(source="fitted")
    from fwmav_sim.vehicle.schema import AeroModelSpec

    model = QuasiSteadyBladeElement(AeroModelSpec(
        name="quasi_steady_blade_element",
        coefficients={k: Parameter(value=v, unit="1", provenance=prov)
                      for k, v in coefs.items()}))
    elements = {w.name: discretize(w, vehicle.blade_elements.n_elements,
                                   vehicle.blade_elements.discretization)
                for w in vehicle.wings}
    kin_map = (kinematics if isinstance(kinematics, dict)
               else {w.name: kinematics for w in vehicle.wings})
    out = np.zeros((len(t), 3))
    for i, ti in enumerate(t):
        for w in vehicle.wings:
            for el in elements[w.name]:
                st = element_state(w, el, kin_map[w.name], ti, nu=env.nu,
                                   v_air_W=env.v_air_W)
                out[i] += model.element_forces(st, env).resultant
    return out


def fit_coefficients(vehicle, kinematics, env: Environment,
                     t: np.ndarray, F_meas: np.ndarray,
                     free: tuple[str, ...] = DEFAULT_FREE,
                     dataset_ref: str | None = None) -> SysidResult:
    """Least-squares fit of the free coefficients to measured forces."""
    c0 = {k: vehicle.aero.coefficients[k].value for k in REQUIRED_COEFFICIENTS}
    missing = [k for k in free if k not in REQUIRED_COEFFICIENTS]
    if missing:
        raise ValueError(f"unknown coefficient names in free set: {missing}")
    x0 = np.array([c0[k] for k in free])
    t = np.asarray(t, dtype=float)
    F_meas = np.asarray(F_meas, dtype=float)

    def resid(x):
        return (_simulate_forces(vehicle, kinematics, env, x, free, c0, t)
                - F_meas).ravel()

    base = float(np.sqrt(np.mean(resid(x0) ** 2)))
    sol = least_squares(resid, x0)
    coefs = dict(c0)
    for k, v in zip(free, sol.x, strict=True):
        coefs[k] = float(v)
    return SysidResult(coefficients=coefs, free=tuple(free),
                       rms_residual=float(np.sqrt(np.mean(resid(sol.x) ** 2))),
                       rms_baseline=base, n_samples=len(t))


def fitted_vehicle_fragment(result: SysidResult, dataset_ref: str) -> dict:
    """Vehicle-YAML aero fragment with provenance source: fitted (brief §5.8)."""
    prov = {"source": "fitted",
            "reference": f"sysid fit against dataset {dataset_ref} (M6 Task 9)",
            "uncertainty_or_range": "scale parameters only by default; shape "
                                    "parameters held at prior (see analysis/sysid.py)"}
    return {
        "name": "quasi_steady_blade_element",
        "coefficients": {k: {"value": v, "unit": "1", "provenance": dict(prov)}
                         for k, v in result.coefficients.items()},
    }


def load_fit_dataset(dataset_dir: str | Path, force_columns: tuple[str, str, str]):
    """Load an M4 dataset's force columns as (t, F, manifest).

    Columns in the manifest are a dict keyed by column name; vector channels
    follow the compare-module convention of one column per component
    (e.g. wing_right_F_total_x/y/z).
    """
    dataset_dir = Path(dataset_dir)
    import pandas as pd

    manifest = yaml.safe_load((dataset_dir / "manifest.yaml").read_text(encoding="utf-8"))
    label = manifest.get("label")
    if label != "calibration":
        raise ValueError(
            f"dataset label must be 'calibration' for fitting (got {label!r}); "
            "a dataset used for fitting can never afterwards be validation (M4 rule)")
    csvs = sorted(dataset_dir.glob("*.csv"))
    if not csvs:
        raise ValueError(f"no CSV in {dataset_dir}")
    df = pd.read_csv(csvs[0])
    col_specs = manifest.get("columns", {})
    from fwmav_sim.datasets.importer import _UNIT_TO_SI

    def col(name):
        factor = _UNIT_TO_SI.get(col_specs.get(name, {}).get("unit", "1"), 1.0)
        return df[name].to_numpy() * factor

    t = col("t")
    F = np.stack([col(c) for c in force_columns], axis=1)
    return t, F, manifest


def write_fragment(result: SysidResult, dataset_ref: str, out: str | Path) -> Path:
    out = Path(out)
    fragment = fitted_vehicle_fragment(result, dataset_ref)
    out.write_text(yaml.safe_dump(fragment) +
                   f"\n# sysid: rms {result.rms_baseline:.4g} -> "
                   f"{result.rms_residual:.4g} over {result.n_samples} samples "
                   f"(free: {', '.join(result.free)})\n", encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps({"rms_baseline": result.rms_baseline,
                                     "rms_residual": result.rms_residual,
                                     "free": list(result.free),
                                     "dataset": dataset_ref}, indent=2),
                         encoding="utf-8")
    return out
