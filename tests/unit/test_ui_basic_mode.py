"""GUI basic mode tests (M6.5 Task 5): default window hides advanced docks;
the toggle reveals them; the core loop stays visible."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_default_is_basic_mode(qapp, tmp_path):
    from fwmav_sim.ui.main_window import MainWindow

    win = MainWindow(experiments_dir=tmp_path)
    assert not win.advanced_button.isChecked()
    for dock in win._advanced_docks:
        assert not dock.isVisible()


def test_toggle_reveals_and_hides(qapp, tmp_path):
    from fwmav_sim.ui.main_window import MainWindow

    win = MainWindow(experiments_dir=tmp_path)
    win.advanced_button.setChecked(True)
    for dock in win._advanced_docks:
        assert not dock.isHidden()  # explicit hide state (offscreen-safe)
    win.advanced_button.setChecked(False)
    for dock in win._advanced_docks:
        assert dock.isHidden()
    # the run panel's dock is untouched by the toggle (core loop visible)
    assert not win.run_panel.isHidden()
