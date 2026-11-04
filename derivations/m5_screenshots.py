"""Capture offscreen screenshots of the workbench panels for the M5 report.

Qt offscreen platform supports QWidget.grab() without a display. Evidence
artefacts for docs/milestone_reports/M5.md (§10 M5: screenshot set).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

OUT = Path("docs/milestone_reports/m5_screenshots")


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    from fwmav_sim.ui.main_window import MainWindow

    OUT.mkdir(parents=True, exist_ok=True)
    win = MainWindow(Path("experiments"))
    win.resize(1280, 800)
    win.show()
    app.processEvents()

    win.load_experiment(Path("experiments/m1_fixed_body_hover.yaml"))
    win._on_inspector_select(("body", "thorax"))
    app.processEvents()
    win.grab().save(str(OUT / "01_main_inspector_editor.png"))
    win.inspector.grab().save(str(OUT / "02_inspector_provenance_badges.png"))
    win.editor.grab().save(str(OUT / "03_parameter_editor.png"))
    win.run_panel.grab().save(str(OUT / "04_run_panel.png"))
    win.validity.grab().save(str(OUT / "05_validity_panel.png"))
    print(f"screenshots -> {OUT}")


if __name__ == "__main__":
    main()
