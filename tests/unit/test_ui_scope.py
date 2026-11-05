"""Scope panel tests (M6 Task 3): checkbox overlays, cursor linkage, 3D seek."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fwmav_sim.core.timeseries import TimeSeries


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _ts(period=0.1):
    t = np.linspace(0, 0.3, 301)
    return TimeSeries(
        t=t,
        channels={
            "wing_right/F_total": np.stack([np.sin(2 * np.pi * 10 * t),
                                            np.cos(2 * np.pi * 10 * t),
                                            -0.5 * np.ones_like(t)], axis=1),
            "wing_right/P_aero": np.abs(np.sin(2 * np.pi * 10 * t)),
            "z": np.sin(2 * np.pi * t),
        },
        meta={"kind": "fixed_body", "wingbeat_period_s": period},
    )


def test_checkbox_overlays_and_default_check(qapp):
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    panel = TimeSeriesPanel()
    panel.set_record(_ts())
    assert panel._checked() == ["wing_right/F_total"]  # default first channel
    panel._items["wing_right/P_aero"].setCheckState(0, Qt.CheckState.Checked)
    assert panel._checked() == ["wing_right/F_total", "wing_right/P_aero"]
    assert panel.plot.listDataItems()  # plotted without error


def test_cursor_signal_and_3d_seek(qapp):
    from fwmav_sim.ui.scene3d import Scene3DPanel
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    ts = _ts()
    panel = TimeSeriesPanel()
    panel.set_record(ts)
    seen = []
    panel.cursor_moved.connect(seen.append)
    panel.set_cursor(0.123)
    assert seen == [pytest.approx(0.123)]

    # 3D seek: free_body replay seeks nearest recorded frame
    scene = Scene3DPanel()
    scene._vis = object()
    ctx = type("C", (), {"ts": ts})()
    scene._replay = ("free_body", ctx, len(ts.t))
    frames = []
    scene._render_frame = lambda i: frames.append(i)
    scene.seek_time(0.123)
    assert frames == [int(np.argmin(np.abs(ts.t - 0.123)))]


def test_phase_average_overlay(qapp):
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    panel = TimeSeriesPanel()
    panel.set_record(_ts())
    assert panel.avg_toggle.isEnabled()
    panel.avg_toggle.setChecked(True)
    # 1 channel (3 components) + 0 scalar-avg (vector channel skipped)
    panel._items["wing_right/P_aero"].setCheckState(0, Qt.CheckState.Checked)
    names = panel.plot.listDataItems()
    assert names  # scalar channel got its dashed avg overlay without error


def test_plotted_buffer_matches_record(qapp):
    """Freshness contract (brief §10 M5): displayed arrays equal the record."""
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    ts = _ts()
    panel = TimeSeriesPanel()
    panel.set_record(ts)
    t, y, name = panel.plotted
    np.testing.assert_array_equal(y, ts[name])
    np.testing.assert_array_equal(t, ts.t)
