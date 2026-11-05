"""Run control panel (M5 Task 6): experiment selection, run/stop in a worker
thread, progress, and record-open on completion. Runs go through the same
simulation.run.run_experiment the CLI uses — the UI adds zero new physics
(brief §3.1).
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


class RunWorker(QThread if _HAVE_QT else object):
    """Runs an experiment off the UI thread; reports the record dir."""

    finished_ok = Signal(str) if _HAVE_QT else None
    failed = Signal(str) if _HAVE_QT else None

    def __init__(self, experiment_path: Path, out_root: Path, parent=None):
        super().__init__(parent)
        self._experiment = experiment_path
        self._out = out_root

    def run(self):  # noqa: D102
        try:
            from fwmav_sim.simulation.run import run_experiment

            record = run_experiment(self._experiment, self._out)
            self.finished_ok.emit(str(record))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class RunPanel(QWidget if _HAVE_QT else object):
    """Experiment picker + run button + status."""

    def __init__(self, experiments_dir: Path, on_record_ready, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the run panel")
        super().__init__(parent)
        self._on_record_ready = on_record_ready
        self._worker: RunWorker | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("experiment:"))
        self.combo = QComboBox()
        self.combo.setToolTip(
            "Experiment YAML to run: wires a vehicle + wing kinematics +\n"
            "environment + solver settings into one run (file → open\n"
            "experiment… also loads its vehicle into the inspector/editor).")
        self.refresh(experiments_dir)
        layout.addWidget(self.combo)
        self.run_button = QPushButton("run")
        self.run_button.setToolTip(
            "Run the selected experiment in a worker thread (same code path as\n"
            "`fwmav-sim run`). On completion the record opens in the time-series,\n"
            "3D and validity views automatically.")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate while running
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel("idle")
        layout.addWidget(self.status)

    def refresh(self, experiments_dir: Path) -> None:
        self.combo.clear()
        for p in sorted(Path(experiments_dir).rglob("*.yaml")):
            self.combo.addItem(str(p))

    def current_experiment(self) -> Path:
        return Path(self.combo.currentText())

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return  # one run at a time
        self.run_button.setEnabled(False)
        self.progress.show()
        self.status.setText("running…")
        self._worker = RunWorker(self.current_experiment(), Path("outputs"))
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _done(self, record_dir: str):
        self.run_button.setEnabled(True)
        self.progress.hide()
        self.status.setText(f"done: {record_dir}")
        self._on_record_ready(Path(record_dir))

    def _failed(self, message: str):
        self.run_button.setEnabled(True)
        self.progress.hide()
        self.status.setText(f"FAILED: {message}")
