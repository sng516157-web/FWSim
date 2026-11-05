"""Workflow ribbon (M6 Task 2): the golden path as a toolstrip.

Five stages — Load, Inspect/Edit, Run, Results, Validity — always visible,
each a button that raises its panel; the current stage highlights so a new
user can see where they are in the loop (MATLAB toolstrip inspiration).
The strip is a navigator only: it adds no behaviour of its own.
"""

from __future__ import annotations

try:
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

STAGES = ("load", "inspect", "run", "results", "validity")

_ACTIVE = "QPushButton { font-weight: bold; background: #3A5F8A; color: white; }"


class WorkflowRibbon(QWidget if _HAVE_QT else object):
    """Stage buttons + current-stage highlighting."""

    def __init__(self, actions: dict[str, callable], parent=None):
        """actions: stage name -> zero-arg callable invoked by its button."""
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the workflow ribbon")
        super().__init__(parent)
        self._buttons: dict[str, object] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self._hint = QLabel("workflow:")
        layout.addWidget(self._hint)
        for stage in STAGES:
            btn = QPushButton(stage.capitalize())
            btn.setCheckable(True)
            btn.setToolTip({
                "load": "Open an experiment YAML (vehicle + kinematics + solver recipe)",
                "inspect": "Inspect/edit the vehicle tree and parameters (left dock)",
                "run": "Pick the experiment and run it (right dock)",
                "results": "Time-series scope and 3D view of the loaded record",
                "validity": "Assumptions in force and regime warnings for the run",
            }[stage])
            if stage in actions:
                # clicked(bool) emits the checked state; actions take no args
                btn.clicked.connect(lambda _c=False, f=actions[stage]: f())
            layout.addWidget(btn)
            self._buttons[stage] = btn
        layout.addStretch()
        self.set_stage("load")

    def set_stage(self, stage: str) -> None:
        """Highlight the current stage (called by the main window on state changes)."""
        for name, btn in self._buttons.items():
            btn.setChecked(name == stage)
            btn.setStyleSheet(_ACTIVE if name == stage else "")
