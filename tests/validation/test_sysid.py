"""Sysid fitting tests (M6 Task 9): synthetic known-coefficient recovery."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.interface import Environment
from fwmav_sim.analysis.sysid import (
    DEFAULT_FREE,
    fit_coefficients,
    fitted_vehicle_fragment,
    load_fit_dataset,
)
from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.vehicle.schema import AeroModelSpec, Parameter, Provenance

_ENV = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))

# truth: a perturbed coefficient set the fit must recover
_TRUTH = {"cl_0": 0.30, "cl_slope": 1.35, "cd_0": 1.70, "cd_slope": 1.80,
          "cl_freq": 2.13, "cl_phase_deg": 7.2, "cd_freq": 2.04, "cd_phase_deg": 9.82}


def _vehicle_with(coefs, base="reference_vehicles/airpulse.yaml"):
    vehicle = load_vehicle(base)
    aero = AeroModelSpec(
        name="quasi_steady_blade_element",
        coefficients={k: Parameter(value=v, unit="1",
                                   provenance=Provenance(source="estimated"))
                      for k, v in coefs.items()})
    return vehicle.model_copy(update={"aero": aero})


def test_fit_recovers_known_coefficients():
    """Synthetic proof: forces generated with a known set are recovered to
    < 5% from a perturbed start (the informative scale parameters)."""
    from fwmav_sim.analysis.sysid import _simulate_forces

    exp = load_experiment("experiments/m4_airpulse_hover_fixed.yaml")
    kin = exp.kinematics
    v_truth = _vehicle_with(_TRUTH)
    # start the fit from the stock Dickinson values (airpulse.yaml)
    v_start = load_vehicle("reference_vehicles/airpulse.yaml")
    f = kin.flap_frequency.value
    t = np.arange(0.0, 2.0 / f, 1.0 / (f * 60))  # 2 wingbeats, 60 steps/beat
    F_meas = _simulate_forces(v_truth, kin, _ENV,
                              np.array([_TRUTH[k] for k in DEFAULT_FREE]),
                              DEFAULT_FREE, _TRUTH, t)

    # noiseless: tight recovery bound (identifiability of the fit form itself)
    result = fit_coefficients(v_start, kin, _ENV, t, F_meas,
                              dataset_ref="synthetic_test")
    for k in DEFAULT_FREE:
        assert result.coefficients[k] == pytest.approx(_TRUTH[k], rel=0.02), k
    assert result.rms_residual < result.rms_baseline * 0.05

    # with 0.5% measurement noise: recovery degrades gracefully (< 10%)
    rng = np.random.default_rng(0)
    F_noisy = F_meas + rng.normal(0.0, 0.005 * np.abs(F_meas).max(), F_meas.shape)
    result_n = fit_coefficients(v_start, kin, _ENV, t, F_noisy,
                                dataset_ref="synthetic_test")
    for k in DEFAULT_FREE:
        assert result_n.coefficients[k] == pytest.approx(_TRUTH[k], rel=0.10), k

    frag = fitted_vehicle_fragment(result, "synthetic_test")
    assert frag["coefficients"]["cl_0"]["provenance"]["source"] == "fitted"
    assert "synthetic_test" in frag["coefficients"]["cl_0"]["provenance"]["reference"]


def test_calibration_label_enforced(tmp_path):

    d = tmp_path / "ds"
    d.mkdir()
    (d / "manifest.yaml").write_text(
        "label: validation\nname: x\ncolumns:\n  t:\n    unit: s\n"
        "    quantity: time\n    frame: W\n", encoding="utf-8")
    (d / "data.csv").write_text("t,fx\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        load_fit_dataset(d, ("fx", "fy", "fz"))
