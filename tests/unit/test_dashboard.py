"""Dashboard write-through test (M9 T4, offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from fwmav_sim.control.pipeline import BlockInstance, Pipeline  # noqa: E402
from fwmav_sim.ui.workbench import DashboardPanel  # noqa: E402


class _FakePlant:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class _FakeSession:
    def __init__(self, pipeline):
        self.plant = _FakePlant(pipeline)


def _pipeline():
    return Pipeline([BlockInstance.model_validate({
        "id": "pid", "type": "controller_pid",
        "params": {"state_idx": 0, "setpoint": 0.0, "kp": 1.0}, "inputs": []})])


def test_dashboard_populates_and_writes():
    app = QApplication.instance() or QApplication([])
    pipe = _pipeline()
    dash = DashboardPanel()
    dash.bind(_FakeSession(pipe))
    assert dash._rows, "dashboard should list the PID's live params"
    rows = {name: (b, s, l, rng) for b, name, s, l, rng in dash._rows}
    block_id, slider, lab, (lo, hi) = rows["kp"][0], rows["kp"][1], rows["kp"][2], rows["kp"][3]
    assert block_id == "pid"
    slider.setValue(1000)  # max of the range
    assert pipe.live_params()["pid"]["kp"] == pytest.approx(hi)


def test_dashboard_no_pipeline_message():
    app = QApplication.instance() or QApplication([])
    class _NoPipe:
        pipeline = None
    class _Sess:
        plant = _NoPipe()
    dash = DashboardPanel()
    dash.bind(_Sess())
    assert dash._layout.count() >= 1  # message label, no crash
