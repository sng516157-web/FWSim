"""M4 pipeline proof: synthetic known-truth dataset round trip.

A run is generated in-test (fixed-body, small), converted into a synthetic
dataset with known phase offset and noise, and the compare pipeline must
recover alignment and residuals within the known noise level. Labelled
synthetic; never a validation claim (brief §14.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.analysis.compare import compare
from fwmav_sim.datasets.synthetic import make_synthetic_dataset
from fwmav_sim.environment import still_air
from fwmav_sim.experiment.config import load_vehicle
from fwmav_sim.experiment.record import save_record
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.simulation.fixed_body import run_fixed_body

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic pipeline test"}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV}


def test_synthetic_dataset_roundtrip(tmp_path):
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    kin = PrescribedWingKinematics(
        flap_amplitude=_param(1.0, "rad"), flap_frequency=_param(15.0, "Hz"),
        pitch_amplitude=_param(0.7, "rad"), pitch_phase=_param(np.pi, "rad"),
        pitch_offset=_param(0.0, "rad"), stroke_plane_angle=_param(0.0, "rad"),
    )
    env = still_air()
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body(vehicle, kin, env, model, n_wingbeats=6, steps_per_wingbeat=100)
    rec = save_record(tmp_path / "out", "synth_source", ts,
                      Path("experiments/m1_fixed_body_hover.yaml").resolve(),
                      Path("reference_vehicles/bio_abdomen_fwmav.yaml").resolve())

    known_offset = 0.011
    noise = 0.02
    ds_dir = make_synthetic_dataset(rec, tmp_path / "ds", "wing_right/F_total",
                                    component=2, phase_offset_s=known_offset,
                                    noise_std=noise, downsample_hz=222.0, seed=7)
    report_dir = compare(rec, ds_dir, "wing_right/F_total", component=2,
                         out_root=tmp_path / "cmp")
    report = json.loads((report_dir / "comparison_report.json").read_text())

    # alignment recovers the known offset: the stage-file's < 0.1%-of-wingbeat
    # bound applies to the clean unit-test sinusoid (tests/unit/test_datasets.py);
    # this roundtrip uses a short, noisy, harmonic-rich window, for which the
    # template estimator's measured accuracy is ~0.3% of a wingbeat (documented
    # in the M4 report).
    assert abs(report["alignment"]["phase_offset_s"] - known_offset) < 0.005 / 15.0
    # residuals within the known injected noise level (2% rel -> ~0.05 NRMSE here)
    assert report["metrics"]["nrmse"] < 0.10
    assert report["metrics"]["pearson"] > 0.99
    # honest labels
    assert report["validation_status"] == "SYNTHETIC_PROOF"
    assert report["dataset_synthetic"] is True
    assert report["dataset_label"] == "calibration"
