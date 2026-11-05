"""PySide6 application shell (M5 Task 1; DR-002).

Single process: the UI imports fwmav_sim directly — no server, no IPC.
PySide6 is an optional dependency (pip install fwmav-sim[gui]); imports are
guarded so headless installs keep working.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The GUI requires the gui extra: pip install fwmav-sim[gui] "
              "(PySide6, pyqtgraph)")
        return 1

    # MeshCat's QWebEngineView on Windows prefers these attributes set early
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
    # Qt-documented requirement for QWebEngineView alongside other widgets in a
    # resizable window; without it the 3D view can glitch black on maximize /
    # fullscreen transitions
    from PySide6.QtCore import Qt

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("FlapSim — FWMAV Workbench")

    from fwmav_sim.ui.main_window import MainWindow

    win = MainWindow()
    win.show()
    return app.exec()
