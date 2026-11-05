"""Console dock tests (M6 Task 1): failures must surface, never vanish."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import logging
import sys

import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_console_captures_logging_and_exceptions(qapp):
    from fwmav_sim.ui.console import ConsolePanel

    panel = ConsolePanel()
    panel.install_global_hooks()
    logging.getLogger().warning("console-test warning")
    assert any("console-test warning" in m and lvl == "warning"
               for lvl, m in panel.entries)

    class _Exc(Exception):
        pass

    sys.excepthook(_Exc, _Exc("boom"), None)
    assert any(lvl == "error" and "boom" in m for lvl, m in panel.entries)


def test_report_failure_logs_and_dialogs(qapp, monkeypatch):
    from fwmav_sim.ui.console import ConsolePanel

    panel = ConsolePanel()
    shown = []
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical",
                        staticmethod(lambda *a, **k: shown.append(a)))
    panel.report_failure("run failed", "solver diverged at t=0.01")
    assert any(lvl == "error" and "solver diverged" in m for lvl, m in panel.entries)
    assert shown  # modal dialog fired


def test_run_panel_failure_routes_to_callback(qapp):
    from fwmav_sim.ui.run_panel import RunPanel

    failures = []
    panel = RunPanel("experiments", on_record_ready=lambda p: None,
                     on_failure=failures.append)
    panel._failed("ValueError: bad experiment")
    assert failures == ["ValueError: bad experiment"]
    assert "FAILED" in panel.status.text()
