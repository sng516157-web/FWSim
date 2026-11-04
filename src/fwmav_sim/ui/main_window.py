"""Main window (M5 Task 7): dockable panels wired into the brief §9 reference
workflow — select vehicle → inspect → configure → fixed-body run → inspect
forces → enable abdomen → suspended test → compare → free-body run →
save/reload → headless repeat.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDockWidget,
        QFileDialog,
        QMainWindow,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

import yaml

from fwmav_sim.experiment.config import load_experiment, load_vehicle


class MainWindow(QMainWindow if _HAVE_QT else object):
    """FlapSim workbench shell."""

    def __init__(self, experiments_dir: Path | None = None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the workbench")
        super().__init__()
        self.setWindowTitle("FlapSim — FWMAV Workbench")
        self._experiment_path: Path | None = None
        self._vehicle = None
        self._record_dir: Path | None = None

        from fwmav_sim.ui.inspector import InspectorPanel
        from fwmav_sim.ui.parameter_editor import ParameterEditor
        from fwmav_sim.ui.run_panel import RunPanel
        from fwmav_sim.ui.scene3d import Scene3DPanel
        from fwmav_sim.ui.timeseries import TimeSeriesPanel
        from fwmav_sim.ui.validity_panel import ValidityPanel

        # centre: 3D over time-series tabs
        self.scene3d = Scene3DPanel()
        self.timeseries = TimeSeriesPanel()
        centre = QTabWidget()
        centre.addTab(self.scene3d, "3D")
        centre.addTab(self.timeseries, "time series")
        self.setCentralWidget(centre)

        # left dock: inspector + editor
        self.inspector = InspectorPanel(self._on_inspector_select)
        self.editor = ParameterEditor(self._on_param_change)
        left = QTabWidget()
        left.addTab(self.inspector, "inspector")
        left.addTab(self.editor, "parameters")
        dock_l = QDockWidget("vehicle", self)
        dock_l.setWidget(left)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_l)

        # right dock: run control + validity
        self.run_panel = RunPanel(experiments_dir or Path("experiments"), self._on_record)
        self.validity = ValidityPanel()
        right = QVBoxLayout()
        right.addWidget(self.run_panel)
        right.addWidget(self.validity)
        right_w = QWidget()
        right_w.setLayout(right)
        dock_r = QDockWidget("run / validity", self)
        dock_r.setWidget(right_w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_r)

        # menu
        menu = self.menuBar().addMenu("file")
        for label, slot in (
            ("open experiment…", self._open_experiment),
            ("save experiment as…", self._save_experiment_as),
            ("open run record…", self._open_record),
            ("compare with dataset…", self._compare),
            ("quit", self.close),
        ):
            menu.addAction(label, slot)
        self.statusBar().showMessage("open an experiment to begin (file → open experiment…)")

    # --- workflow wiring (brief §9) -------------------------------------------

    def _open_experiment(self):
        path, _ = QFileDialog.getOpenFileName(self, "open experiment", "experiments",
                                          "YAML (*.yaml)")
        if path:
            self.load_experiment(Path(path))

    def load_experiment(self, path: Path) -> None:
        """Steps 1-3: select + inspect + configure."""
        exp = load_experiment(path)
        self._experiment_path = path
        if exp.vehicle:
            self._vehicle = load_vehicle(path.parent / exp.vehicle)
            self.inspector.set_vehicle(self._vehicle)
        self.run_panel.status.setText(f"loaded: {exp.name}")
        self.statusBar().showMessage(f"experiment: {exp.name} ({exp.kind})")

    def _on_inspector_select(self, key):
        """Inspector selection drives the parameter editor."""
        if self._vehicle is None:
            return
        kind, name = key
        if kind == "body":
            self.editor.set_target(next(b for b in self._vehicle.bodies if b.name == name))
        elif kind == "wing":
            self.editor.set_target(next(w for w in self._vehicle.wings if w.name == name))

    def _on_param_change(self, _updated):
        self.statusBar().showMessage("edited (unsaved) — file → save experiment as…")

    def _save_experiment_as(self):
        if self._vehicle is None or self._experiment_path is None:
            self.statusBar().showMessage("nothing to save")
            return
        path, _ = QFileDialog.getSaveFileName(self, "save vehicle YAML",
                                              str(self._experiment_path.parent),
                                              "YAML (*.yaml)")
        if path:
            Path(path).write_text(yaml.safe_dump(self._vehicle.model_dump()), encoding="utf-8")
            self.statusBar().showMessage(f"vehicle saved: {path}")

    def _on_record(self, record_dir: Path):
        """Steps 4-5 / 7-10: run finished — open the record for inspection."""
        from fwmav_sim.experiment.record import load_record

        self._record_dir = record_dir
        ts, _meta = load_record(record_dir)
        self.timeseries.set_record(ts)
        self.validity.show_record(record_dir)
        self.scene3d.show_record_frame(record_dir, frame=0)
        self.statusBar().showMessage(f"record: {record_dir.name}")

    def _open_record(self):
        path = QFileDialog.getExistingDirectory(self, "open run record", "outputs")
        if path:
            self._on_record(Path(path))

    def _compare(self):
        """Step 8: compare the current record against a dataset (M4)."""
        if self._record_dir is None:
            self.statusBar().showMessage("open or run a record first")
            return
        ds = QFileDialog.getExistingDirectory(self, "dataset directory", "datasets")
        if not ds:
            return
        from fwmav_sim.analysis.compare import compare

        channel = self.timeseries.plotted[2] if self.timeseries.plotted else None
        if channel is None:
            self.statusBar().showMessage("select a channel in the time-series view first")
            return
        report = compare(self._record_dir, Path(ds), channel, out_root=Path("outputs"))
        self.statusBar().showMessage(f"comparison report: {report}")
