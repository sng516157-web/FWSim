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
    """FWSim workbench shell."""

    def __init__(self, experiments_dir: Path | None = None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the workbench")
        super().__init__()
        self.setWindowTitle("FWSim — FWMAV Workbench")
        self._experiment_path: Path | None = None
        self._vehicle = None
        self._record_dir: Path | None = None

        from fwmav_sim.ui.console import ConsolePanel
        from fwmav_sim.ui.inspector import InspectorPanel
        from fwmav_sim.ui.parameter_editor import ParameterEditor
        from fwmav_sim.ui.run_panel import RunPanel
        from fwmav_sim.ui.scene3d import Scene3DPanel
        from fwmav_sim.ui.timeseries import TimeSeriesPanel
        from fwmav_sim.ui.validity_panel import ValidityPanel

        self.console = ConsolePanel()
        self.console.install_global_hooks()

        # centre: 3D over time-series tabs
        self.scene3d = Scene3DPanel()
        self.timeseries = TimeSeriesPanel()
        centre = QTabWidget()
        centre.addTab(self.scene3d, "3D")
        centre.addTab(self.timeseries, "time series")
        self.setCentralWidget(centre)
        # scope cursor scrubs the 3D frame (M6 Task 3)
        self.timeseries.cursor_moved.connect(self.scene3d.seek_time)

        # left dock: inspector + editor + records browser (M6 T5)
        from fwmav_sim.ui.records import RecordsPanel

        self.inspector = InspectorPanel(self._on_inspector_select)
        self.editor = ParameterEditor(self._on_param_change)
        self.records = RecordsPanel(Path("outputs"), self._on_record, self._on_compare_record)
        left = QTabWidget()
        left.addTab(self.inspector, "inspector")
        left.addTab(self.editor, "parameters")
        left.addTab(self.records, "records")
        dock_l = QDockWidget("vehicle", self)
        dock_l.setWidget(left)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_l)

        # right dock: run control + validity
        self.run_panel = RunPanel(experiments_dir or Path("experiments"), self._on_record,
                                  on_failure=self._on_run_failure)
        self.validity = ValidityPanel()
        right = QVBoxLayout()
        right.addWidget(self.run_panel)
        right.addWidget(self.validity)
        right_w = QWidget()
        right_w.setLayout(right)
        dock_r = QDockWidget("run / validity", self)
        dock_r.setWidget(right_w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_r)

        # bottom dock: console (M6 Task 1 — failures must be visible)
        dock_b = QDockWidget("console", self)
        dock_b.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_b)

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

        # workflow ribbon (M6 Task 2): the golden path as a toolstrip
        from fwmav_sim.ui.workflow import WorkflowRibbon

        self.ribbon = WorkflowRibbon({
            "load": self._open_experiment,
            "inspect": dock_l.show,
            "run": dock_r.show,
            "results": lambda: centre.setCurrentWidget(self.timeseries),
            "validity": dock_r.show,
        })
        ribbon_bar = self.addToolBar("workflow")
        ribbon_bar.setMovable(False)
        ribbon_bar.addWidget(self.ribbon)

        self.statusBar().showMessage("open an experiment to begin (file → open experiment…)")

    # --- workflow wiring (brief §9) -------------------------------------------

    def _open_experiment(self):
        path, _ = QFileDialog.getOpenFileName(self, "open experiment", "experiments",
                                          "YAML (*.yaml)")
        if path:
            self.load_experiment(Path(path))

    def load_experiment(self, path: Path) -> None:
        """Steps 1-3: select + inspect + configure; also preview in 3D."""
        exp = load_experiment(path)
        self._experiment = exp
        self._experiment_path = path
        if exp.vehicle:
            self._vehicle = load_vehicle(path.parent / exp.vehicle)
            self.inspector.set_vehicle(self._vehicle)
            if exp.kinematics is not None:
                self.inspector.set_experiment(exp)
        if exp.kind in ("fixed_body", "free_body") and exp.kinematics is not None:
            self.scene3d.set_experiment(path)
        self.run_panel.status.setText(f"loaded: {exp.name}")
        self.console.log("info", f"experiment loaded: {exp.name} ({exp.kind}) — {path}")
        self.ribbon.set_stage("inspect")
        self.statusBar().showMessage(f"experiment: {exp.name} ({exp.kind})")

    def _on_run_failure(self, message: str):
        self.console.report_failure("run failed", message)

    def _on_inspector_select(self, key):
        """Inspector selection drives the parameter editor."""
        kind, name = key
        if kind == "kinematics":
            kin = self._experiment.kinematics
            target = kin if not isinstance(kin, dict) else kin.get(name)
            if target is not None:
                self.editor.set_target(target)
            return
        if self._vehicle is None:
            return
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
        self.scene3d.set_record(record_dir)
        self.console.log("info", f"record opened: {record_dir.name}")
        self.ribbon.set_stage("results")
        self.records.refresh()
        self.statusBar().showMessage(f"record: {record_dir.name}")

    def _on_compare_record(self, record_dir: Path):
        """Overlay another record on the scope (M6 Task 5)."""
        if self._record_dir is None:
            self.statusBar().showMessage("open a record first, then compare")
            return
        from fwmav_sim.experiment.record import load_record

        ts2, _ = load_record(record_dir)
        self.timeseries.set_overlay(ts2, Path(record_dir).name[:15])
        self.console.log("info", f"comparing: {self._record_dir.name} vs {Path(record_dir).name}")
        self.ribbon.set_stage("results")

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
