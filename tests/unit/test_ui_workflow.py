"""Workflow ribbon tests (M6 Task 2)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_ribbon_stages_and_highlight(qapp):
    from fwmav_sim.ui.workflow import STAGES, WorkflowRibbon

    fired = []
    ribbon = WorkflowRibbon({s: (lambda s=s: fired.append(s)) for s in STAGES})
    assert list(ribbon._buttons) == list(STAGES)
    ribbon._buttons["run"].click()
    assert fired == ["run"]
    ribbon.set_stage("results")
    assert ribbon._buttons["results"].isChecked()
    assert not ribbon._buttons["run"].isChecked()
    assert "bold" in ribbon._buttons["results"].styleSheet()


def test_main_window_stage_transitions(qapp, monkeypatch, tmp_path):
    """Loading an experiment moves the ribbon to inspect; a record to results."""
    from fwmav_sim.ui.main_window import MainWindow

    win = MainWindow(experiments_dir=tmp_path)
    assert win.ribbon._buttons["load"].isChecked()
    # load_experiment without real files: monkeypatch the panel entry points
    monkeypatch.setattr(win.inspector, "set_vehicle", lambda v: None)
    monkeypatch.setattr(win.scene3d, "set_experiment", lambda p: None)
    win.load_experiment("experiments/m1_fixed_body_hover.yaml" and
                        __import__("pathlib").Path("experiments/m1_fixed_body_hover.yaml"))
    assert win.ribbon._buttons["inspect"].isChecked()
