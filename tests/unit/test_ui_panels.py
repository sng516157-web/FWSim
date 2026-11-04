"""§11.1 M5 UI unit tests: schema->form generation round trip, provenance
editing, demo-plot freshness. Offscreen Qt (no display needed).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _vehicle():
    from fwmav_sim.experiment.config import load_vehicle

    return load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")


def test_form_generation_round_trip(qapp):
    """Schema -> form -> edit -> YAML -> reload: values round-trip."""
    from fwmav_sim.ui.parameter_editor import ParameterEditor

    vehicle = _vehicle()
    changes = []
    editor = ParameterEditor(on_change=changes.append)
    body = vehicle.bodies[0]
    editor.set_target(body)
    assert "mass" in editor._fields

    # edit the mass value through the form
    spin = editor._fields["mass"][0]
    spin.setValue(0.0123)
    assert body.mass.value == pytest.approx(0.0123)
    assert changes  # on_change fired

    # YAML round trip of the edited vehicle
    text = yaml.safe_dump(vehicle.model_dump())
    from fwmav_sim.vehicle.schema import VehicleDefinition

    reloaded = VehicleDefinition.model_validate(yaml.safe_load(text))
    assert reloaded.bodies[0].mass.value == pytest.approx(0.0123)


def test_provenance_editing(qapp):
    from fwmav_sim.ui.parameter_editor import ParameterEditor

    vehicle = _vehicle()
    editor = ParameterEditor(on_change=lambda _o: None)
    body = vehicle.bodies[0]
    editor.set_target(body)
    prov_combo = editor._fields["mass"][1]
    prov_combo.setCurrentText("measured")
    assert body.mass.provenance.source == "measured"


def test_form_rejects_bad_value(qapp):
    """Schema validation still fires after edit (negative mass rejected on
    reload — the editor writes the YAML, the schema enforces)."""
    vehicle = _vehicle()
    vehicle.bodies[0].mass.value = -1.0
    from pydantic import ValidationError

    from fwmav_sim.vehicle.schema import VehicleDefinition

    with pytest.raises(ValidationError):
        VehicleDefinition.model_validate(yaml.safe_load(yaml.safe_dump(vehicle.model_dump())))


def test_demo_plots_live(qapp, tmp_path):
    """No stored presentation data (brief §10 M5): the arrays displayed by the
    time-series panel must equal those recomputed fresh from the record."""
    # freshest available record: generate one in-test (small fixed-body run)
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.environment import still_air
    from fwmav_sim.experiment.config import load_vehicle
    from fwmav_sim.experiment.record import load_record, save_record
    from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
    from fwmav_sim.simulation.fixed_body import run_fixed_body
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    prov = {"source": "estimated", "uncertainty_or_range": "test"}

    def p(v, u="m"):
        return {"value": v, "unit": u, "provenance": prov}

    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    kin = PrescribedWingKinematics(
        flap_amplitude=p(1.0, "rad"), flap_frequency=p(15.0, "Hz"),
        pitch_amplitude=p(0.7, "rad"), pitch_phase=p(np.pi, "rad"),
        pitch_offset=p(0.0, "rad"), stroke_plane_angle=p(0.0, "rad"),
    )
    ts = run_fixed_body(vehicle, kin, still_air(), QuasiSteadyBladeElement(vehicle.aero),
                        n_wingbeats=1, steps_per_wingbeat=50)
    rec = save_record(tmp_path, "freshness", ts,
                      Path("experiments/m1_fixed_body_hover.yaml").resolve(),
                      Path("reference_vehicles/bio_abdomen_fwmav.yaml").resolve())
    ts2, _ = load_record(rec)

    panel = TimeSeriesPanel()
    panel.set_record(ts2)
    channel = "wing_right/F_total"
    panel._replot(channel)
    t_disp, y_disp, name = panel.plotted
    assert name == channel
    np.testing.assert_array_equal(y_disp, ts2[channel])  # displayed == record
    # and the record equals the fresh computation by construction (same run)
