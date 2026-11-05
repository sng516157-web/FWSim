"""Validity/assumption panel (M5 Task 6, brief §9): first-class view fed by
the honesty hooks in the run record — assumption IDs in force and regime
warnings from the record's validity section. Never derived by hand.
"""

from __future__ import annotations

try:
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

from fwmav_sim.experiment.record import load_record


class ValidityPanel(QWidget if _HAVE_QT else object):
    """Lists assumptions-in-force and warnings from the current run record."""

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the validity panel")
        super().__init__(parent)
        header = QLabel("assumptions & validity")
        header.setToolTip(
            "Per-run honesty record: which assumptions (A-001…, see\n"
            "docs/assumptions.md) were in force for the loaded run, the run\n"
            "status (OK/FAILED), and regime warnings (e.g. coefficient\n"
            "extrapolation). Read this before trusting any number.")
        layout = QVBoxLayout(self)
        layout.addWidget(header)
        self._label = QLabel("no record loaded — run an experiment or "
                             "file → open run record…")
        self._label.setWordWrap(True)
        layout.addWidget(self._label)
        layout.addStretch()

    def show_record(self, record_dir) -> None:
        _ts, meta = load_record(record_dir)
        validity = meta.get("validity", {})
        assumptions = validity.get("assumption_ids", [])
        warnings = validity.get("warnings", [])
        status = meta.get("status", "OK")
        lines = [f"run status: {status}", "",
                 "assumptions in force:", *[f"  {a}" for a in assumptions]]
        if warnings:
            lines += ["", "warnings:", *[f"  ! {w}" for w in warnings]]
        veh = meta.get("vehicle_file")
        if veh:
            lines += ["", f"vehicle file: {veh}"]
        self._label.setText("\n".join(lines))
