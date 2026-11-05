"""Compact vehicle authoring tests (M6.5 Task 3)."""

from __future__ import annotations

import pytest

from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.vehicle.compact import expand_compact

_COMPACT = """\
format: compact-v1
name: compact_test
provenance: {source: estimated, uncertainty_or_range: "draft"}
bodies:
  - name: thorax
    mass: 0.011
    com_offset: [0.0, 0.0, 0.0]
    inertia: [[2.0e-6, 0.0, 0.0], [0.0, 2.0e-6, 0.0], [0.0, 0.0, 2.0e-6]]
joints: []
wings:
  - name: wing_right
    side: right
    span: 0.12
    root_chord: 0.035
    tip_chord: 0.018
    planform: trapezoidal
    hinge_location: [0.0, 0.005, 0.0]
  - name: wing_left
    side: left
    span: 0.12
    root_chord: 0.035
    tip_chord: 0.018
    planform: trapezoidal
    hinge_location: [0.0, -0.005, 0.0]
aero:
  name: quasi_steady_blade_element
  coefficients:
    cl_0: 0.225
    cl_slope: 1.58
    cl_freq: 2.13
    cl_phase_deg: 7.2
    cd_0: 1.92
    cd_slope: 1.55
    cd_freq: 2.04
    cd_phase_deg: 9.82
"""


def test_compact_loads_and_stamps_provenance(tmp_path):
    f = tmp_path / "v.yaml"
    f.write_text(_COMPACT, encoding="utf-8")
    v = load_vehicle(f)
    assert v.name == "compact_test"
    thorax = v.bodies[0]
    assert thorax.mass.value == pytest.approx(0.011)
    assert thorax.mass.unit == "kg"
    assert thorax.mass.provenance.source == "estimated"
    assert thorax.mass.provenance.uncertainty_or_range == "draft"
    assert v.wings[0].span.unit == "m"
    assert v.aero.coefficients["cl_0"].provenance.source == "estimated"
    assert v.aero.coefficients["cl_phase_deg"].unit == "deg"


def test_compact_equals_expanded_form():
    """The compact dialect expands to exactly what the verbose form means."""
    import yaml

    from fwmav_sim.vehicle.schema import VehicleDefinition

    raw = yaml.safe_load(_COMPACT)
    expanded = expand_compact(raw)
    v = VehicleDefinition.model_validate(expanded)
    # spot-check full nesting materialized
    assert isinstance(expanded["bodies"][0]["mass"], dict)
    assert expanded["bodies"][0]["mass"]["provenance"]["source"] == "estimated"
    assert expanded["wings"][1]["hinge_location"]["value"] == [0.0, -0.005, 0.0]
    assert v.wings[1].pitch_axis_fraction.value == 0.0  # schema default filled


def test_verbose_still_loads():
    """The full provenance format is untouched (M1-M6 compatibility)."""
    v = load_vehicle("reference_vehicles/airpulse.yaml")
    assert v.name == "airpulse"
    assert v.bodies[0].mass.provenance.source == "literature"


def test_compact_experiment_kinematics(tmp_path):
    exp_yaml = """\
format: compact-v1
name: compact_exp
kind: fixed_body
vehicle: v.yaml
kinematics:
  flap_amplitude: 1.0472
  flap_frequency: 15.0
  pitch_amplitude: 0.7854
  pitch_phase: 3.14159
  pitch_offset: 0.0
  stroke_plane_angle: 0.0
environment:
  air_density: 1.225
  kinematic_viscosity: 1.46e-5
  wind: {value: [0.0, 0.0, 0.0], unit: m/s, provenance: {source: estimated}}
n_wingbeats: 1
steps_per_wingbeat: 10
"""
    (tmp_path / "v.yaml").write_text(_COMPACT, encoding="utf-8")
    f = tmp_path / "exp.yaml"
    f.write_text(exp_yaml, encoding="utf-8")
    exp = load_experiment(f)
    assert exp.kinematics.flap_frequency.value == pytest.approx(15.0)
    assert exp.kinematics.flap_frequency.unit == "Hz"
    assert exp.environment.air_density.unit == "kg/m^3"
