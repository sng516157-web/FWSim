"""Sensitivity analysis tests (post-M5 item 2).

Analytic anchors for the quasi-steady fixed-body model: every force term is
exactly linear in air density and exactly quadratic in flapping frequency;
aerodynamic power is cubic in frequency; body mass does not enter fixed-body
forces at all. Central differences recover S = 1, 2, 3, 0 to solver precision.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from fwmav_sim.analysis.sensitivity import (
    default_parameters,
    report_markdown,
    sensitivity_analysis,
)

_PROV = {"source": "estimated"}


def _param(v, unit="1"):
    return {"value": v, "unit": unit, "provenance": dict(_PROV)}


def _write_experiment(tmp_path):
    vehicle = {
        "name": "sens_test",
        "bodies": [{
            "name": "body",
            "mass": _param(0.02, "kg"),
            "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": dict(_PROV)},
            "inertia": {"value": np.diag([5e-5, 5e-5, 5e-5]).tolist(),
                        "unit": "kg m^2", "provenance": dict(_PROV)},
        }],
        "joints": [],
        "wings": [{
            "name": f"wing_{side}", "side": side,
            "span": _param(0.15, "m"),
            "root_chord": _param(0.05, "m"),
            "tip_chord": _param(0.05, "m"),
            "planform": "rectangular",
            "hinge_location": {"value": [0.0, 0.01 if side == "right" else -0.01, 0.0],
                               "unit": "m", "provenance": dict(_PROV)},
            "pitch_axis_fraction": _param(0.0),
        } for side in ("right", "left")],
        "blade_elements": {"n_elements": 8, "discretization": "equal_span"},
        "aero": {"name": "quasi_steady_blade_element",
                 "coefficients": {k: _param(v) for k, v in {
                     "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.2,
                     "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
                 }.items()}},
    }
    experiment = {
        "name": "sens_test_exp",
        "kind": "fixed_body",
        "vehicle": "vehicle.yaml",
        "kinematics": {
            "flap_amplitude": _param(0.9, "rad"),
            "flap_frequency": _param(10.0, "Hz"),
            "pitch_amplitude": _param(0.4, "rad"),
            "pitch_phase": _param(0.8, "rad"),
            "pitch_offset": _param(0.1, "rad"),
            "stroke_plane_angle": _param(0.0, "rad"),
        },
        "environment": {
            "air_density": _param(1.225, "kg/m^3"),
            "kinematic_viscosity": _param(1.5e-5, "m^2/s"),
            "wind": {"value": [0.0, 0.0, 0.0], "unit": "m/s", "provenance": dict(_PROV)},
        },
        "n_wingbeats": 2,
        "steps_per_wingbeat": 60,
    }
    (tmp_path / "vehicle.yaml").write_text(yaml.safe_dump(vehicle), encoding="utf-8")
    exp_path = tmp_path / "experiment.yaml"
    exp_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return exp_path, experiment, vehicle


def test_sensitivity_analytic_exponents(tmp_path):
    exp_path, _, _ = _write_experiment(tmp_path)
    base, entries = sensitivity_analysis(
        exp_path,
        params=["environment:air_density", "kinematics:flap_frequency",
                "vehicle:bodies.body.mass"],
        rel_step=0.05)
    by_path = {e.path: e for e in entries}
    assert all(e.status == "OK" for e in entries)
    # force is exactly linear in rho and quadratic in f; body mass absent
    assert by_path["environment:air_density"].sensitivities["meanFz_total"] \
        == pytest.approx(1.0, abs=1e-8)
    assert by_path["kinematics:flap_frequency"].sensitivities["meanFz_total"] \
        == pytest.approx(2.0, abs=1e-8)
    # power is cubic in frequency; the central difference of a cubic carries
    # the exact truncation bias + rel_step^2
    assert by_path["kinematics:flap_frequency"].sensitivities["wing_right.meanP"] \
        == pytest.approx(3.0 + 0.05**2, abs=1e-8)
    # body mass does not enter fixed-body forces
    assert by_path["vehicle:bodies.body.mass"].sensitivities["meanFz_total"] \
        == pytest.approx(0.0, abs=1e-10)
    # baseline metrics are finite and nonzero
    assert abs(base["meanFz_total"]) > 1e-6


def test_default_parameters_cover_vehicle_and_kinematics(tmp_path):
    exp_path, exp_raw, veh_raw = _write_experiment(tmp_path)
    paths = default_parameters(exp_raw, veh_raw)
    assert "vehicle:wings.wing_right.span" in paths
    assert "vehicle:bodies.body.mass" in paths
    assert "environment:air_density" in paths
    assert "kinematics:flap_frequency" in paths


def test_report_markdown_ranks(tmp_path):
    exp_path, _, _ = _write_experiment(tmp_path)
    base, entries = sensitivity_analysis(
        exp_path, params=["kinematics:flap_frequency", "vehicle:bodies.body.mass"],
        rel_step=0.05)
    md = report_markdown("test", base, entries)
    assert "kinematics:flap_frequency" in md.splitlines()[4]  # ranked first
    assert "| rank |" in md
