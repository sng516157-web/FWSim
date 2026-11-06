"""M9 workbench shell tests (offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from fwmav_sim.ui.workbench import PALETTE, WorkbenchWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_shell_layout(app):
    w = WorkbenchWindow()
    assert w.tabs.count() == 4  # platform | control | 3D live | scopes
    assert w.palette.count() == len(PALETTE)
    assert w.transport.record.isChecked()  # recording on by default (M9)
    assert w.transport.speed_value() == 0.1


def test_palette_adds_control_block(app):
    w = WorkbenchWindow()
    w.tabs.setCurrentWidget(w.pipeline_editor)
    # find the PID entry and double-click it programmatically
    for i in range(w.palette.count()):
        item = w.palette.item(i)
        if item.data(0x0100) == "controller_pid":  # Qt.UserRole
            w._on_palette_add(item)
            break
    scene = w.pipeline_editor.canvas.scene()
    assert any(b["type"] == "controller_pid" for b in scene.graph.blocks.values())


def test_palette_mechanical_redirect_message(app):
    w = WorkbenchWindow()
    w.tabs.setCurrentWidget(w.pipeline_editor)
    for i in range(w.palette.count()):
        item = w.palette.item(i)
        if item.data(0x0100) == "body.box":
            w._on_palette_add(item)
            break
    assert "platform tab" in w.statusBar().currentMessage()


def test_play_requires_experiment(app):
    w = WorkbenchWindow()
    w._on_play(True)
    assert not w.transport.play.isChecked()  # rejected: no experiment
