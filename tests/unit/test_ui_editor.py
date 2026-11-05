"""Parameter editor rework tests (M6 Task 4): matrix editing, inline
validation, per-field reset, experiment kinematics in the inspector."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication

from fwmav_sim.experiment.config import load_vehicle


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _editor(qapp):
    from fwmav_sim.ui.parameter_editor import ParameterEditor

    changes = []
    editor = ParameterEditor(on_change=changes.append)
    return editor, changes


def test_matrix_editing_round_trips(qapp):
    editor, _ = _editor(qapp)
    body = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml").bodies[0]
    editor.set_target(body)
    assert "inertia" in editor._fields  # matrix now editable (was read-only)
    grid = editor._fields["inertia"]
    assert len(grid) == 9
    grid[0].setValue(3.3e-6)  # (0,0)
    grid[1].setValue(1e-8)  # (0,1) — off-diagonal (with its mirror: symmetric)
    grid[3].setValue(1e-8)  # (1,0)
    assert body.inertia.value[0][0] == pytest.approx(3.3e-6)
    assert body.inertia.value[0][1] == pytest.approx(1e-8)  # raw grid position (0,1)
    # YAML round trip of the edited matrix
    from fwmav_sim.vehicle.schema import BodySpec

    reloaded = BodySpec.model_validate(yaml.safe_load(yaml.safe_dump(body.model_dump())))
    assert reloaded.inertia.value[0][0] == pytest.approx(3.3e-6)


def test_inline_validation_flags_invalid(qapp):
    editor, _ = _editor(qapp)
    wing = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml").wings[0]
    editor.set_target(wing)
    spin = editor._fields["span"][0]
    spin.setValue(-1.0)  # violates span > 0
    assert editor._error.text() != ""
    assert spin.styleSheet() != ""
    spin.setValue(0.12)
    assert editor._error.text() == ""
    assert spin.styleSheet() == ""


def test_per_field_reset(qapp):
    editor, _ = _editor(qapp)
    body = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml").bodies[0]
    original = body.mass.value
    editor.set_target(body)
    spin = editor._fields["mass"][0]
    spin.setValue(original * 2)
    editor._reset_field("mass", lambda v: setattr(body, "mass", v))
    assert body.mass.value == pytest.approx(original)


def test_inspector_experiment_section(qapp):
    from fwmav_sim.experiment.config import load_experiment
    from fwmav_sim.ui.inspector import InspectorPanel

    inspector = InspectorPanel(lambda key: None)
    inspector.set_vehicle(load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml"))
    inspector.set_experiment(load_experiment("experiments/m1_fixed_body_hover.yaml"))
    kinds = set(inspector._index.values())
    assert any(k == "kinematics" for k, _ in kinds)
