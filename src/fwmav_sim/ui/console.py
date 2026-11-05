"""Console dock (M6 Task 1): the workbench's command-window equivalent.

Design rule (M6 ground rules): silent failure is a defect class. Three feeds:
- Python ``logging`` records (a Handler attached to the root logger),
- uncaught exceptions in Qt slots (``sys.excepthook``),
- Qt's own message stream (``qInstallMessageHandler`` — includes most
  PySide-internal warnings).

Run-worker failures additionally surface as a modal dialog AND a console
entry with the traceback (wired in main_window / run_panel).
"""

from __future__ import annotations

import html
import logging
import sys
import traceback

try:
    from PySide6.QtCore import qInstallMessageHandler
    from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

_COLORS = {"info": "#C8C8C8", "warning": "#E0B040", "error": "#FF6666"}


class _LogHandler(logging.Handler):
    def __init__(self, emit):
        super().__init__()
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        level = ("error" if record.levelno >= logging.ERROR
                 else "warning" if record.levelno >= logging.WARNING else "info")
        self._emit(level, record.getMessage())


class ConsolePanel(QWidget if _HAVE_QT else object):
    """Colour-coded, always-visible message console."""

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the console panel")
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)
        self.entries: list[tuple[str, str]] = []  # (level, message) — test hook

    def log(self, level: str, message: str) -> None:
        level = level if level in _COLORS else "info"
        self.entries.append((level, message))
        self._text.appendHtml(
            f'<span style="color:{_COLORS[level]}">[{level}] '
            f"{html.escape(message)}</span>")

    # --- app-wide wiring -------------------------------------------------------

    def install_global_hooks(self) -> None:
        """Capture logging, uncaught exceptions and Qt messages."""
        handler = _LogHandler(self.log)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        self._prev_excepthook = sys.excepthook

        def _exc_hook(exc_type, exc, tb):
            self.log("error", "".join(traceback.format_exception(exc_type, exc, tb)).rstrip())

        sys.excepthook = _exc_hook

        def _qt_msg(mode, _ctx, msg):
            self.log("warning" if "warning" in str(mode).lower() else "info", f"Qt: {msg}")

        qInstallMessageHandler(_qt_msg)
        self.log("info", "console ready")

    def report_failure(self, title: str, message: str) -> None:
        """Modal dialog + console entry for a run/action failure."""
        from PySide6.QtWidgets import QMessageBox

        self.log("error", f"{title}: {message}")
        QMessageBox.critical(self, title, message)
