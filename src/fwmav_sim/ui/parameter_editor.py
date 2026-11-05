"""Schema-driven parameter editor (M5 Task 3; reworked M6 Task 4).

Forms are generated from the Pydantic models (never hand-built per field):
every physical Parameter gets value + unit + provenance editing; vectors and
3x3 matrices edit as grids; every edit re-validates the whole object and
flags violations inline (red border + message) instead of at save time;
per-field reset restores the value from when the object was loaded.
Write-back targets the same YAML the CLI consumes — no second config format.
"""

from __future__ import annotations

import copy

try:
    from pydantic import ValidationError
    from PySide6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

from fwmav_sim.vehicle.schema import Matrix3Parameter, Parameter, Vector3Parameter

_SOURCES = ["measured", "literature", "estimated", "fitted"]
_INVALID = "QDoubleSpinBox { border: 1px solid #D04040; }"


class ParameterEditor(QWidget if _HAVE_QT else object):
    """Edit one spec object (BodySpec, WingSpec, ...) with provenance fields.

    Emits on_change(updated_object). The caller is responsible for putting the
    updated object back into the vehicle/experiment and writing YAML.
    """

    def __init__(self, on_change, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the parameter editor")
        super().__init__(parent)
        self._on_change = on_change
        self._target = None
        self._fields: dict[str, list] = {}
        self._original: dict[str, object] = {}
        outer = QVBoxLayout(self)
        self._error = QLabel("")
        self._error.setStyleSheet("color: #D04040")
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._form = QFormLayout(self._inner)
        self._scroll.setWidget(self._inner)
        outer.addWidget(QLabel("select an object in the inspector"))
        outer.addWidget(self._error)
        outer.addWidget(self._scroll)

    def set_target(self, obj) -> None:
        """Build the form for a Pydantic model instance."""
        self._target = obj
        self._fields.clear()
        self._original = {name: copy.deepcopy(getattr(obj, name))
                          for name in type(obj).model_fields}
        while self._form.rowCount():
            self._form.removeRow(0)
        for name in type(obj).model_fields:
            current = getattr(obj, name)
            if isinstance(current, Parameter):
                self._add_parameter_row(name, current)
            elif isinstance(current, Vector3Parameter):
                self._add_vector_row(name, current)
            elif isinstance(current, Matrix3Parameter):
                self._add_matrix_row(name, current)
            else:
                self._add_plain_row(name, current)
        self._error.setText("")

    def _emit(self):
        if self._target is None:
            return
        self._validate()
        self._on_change(self._target)

    def _validate(self) -> bool:
        """Re-validate the whole object; flag violations inline (M6 T4)."""
        try:
            type(self._target).model_validate(self._target.model_dump(mode="python"))
        except ValidationError as exc:
            self._error.setText(exc.errors()[0]["msg"])
            for widgets in self._fields.values():
                for w in widgets:
                    if isinstance(w, QDoubleSpinBox):
                        w.setStyleSheet(_INVALID)
            return False
        self._error.setText("")
        for widgets in self._fields.values():
            for w in widgets:
                if isinstance(w, QDoubleSpinBox):
                    w.setStyleSheet("")
        return True

    def _reset_field(self, name: str, apply_fn) -> None:
        """Restore a field's value from the loaded state."""
        original = self._original.get(name)
        if original is not None:
            apply_fn(copy.deepcopy(original))
            self.set_target(self._target)  # rebuild the form with restored values
            self._emit()

    def _add_parameter_row(self, name: str, param: Parameter) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(9)
        spin.setValue(param.value)
        unit = QLabel(param.unit)
        prov = QComboBox()
        prov.addItems(_SOURCES)
        prov.setCurrentText(param.provenance.source)
        ref = QLineEdit(param.provenance.reference or "")

        def changed(*_):
            param.value = spin.value()
            param.provenance.source = prov.currentText()
            param.provenance.reference = ref.text() or None
            self._emit()

        spin.valueChanged.connect(changed)
        prov.currentTextChanged.connect(changed)
        ref.editingFinished.connect(changed)
        row = QWidget()
        lay = QFormLayout(row)
        lay.addRow("value", spin)
        lay.addRow("unit", unit)
        lay.addRow("source", prov)
        lay.addRow("reference", ref)
        self._form.addRow(name, row)
        self._fields[name] = [spin, prov, ref]
        reset = QPushButton("↺")
        reset.setFixedWidth(28)
        reset.setToolTip("reset to the value at load time")
        reset.clicked.connect(lambda *_: self._reset_field(
            name, lambda v: setattr(self._target, name, v)))
        lay.addRow("", reset)

    def _add_vector_row(self, name: str, param: Vector3Parameter) -> None:
        boxes = []
        row = QWidget()
        lay = QFormLayout(row)
        lay.addRow("unit", QLabel(param.unit))
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(9)
            spin.setValue(param.value[i])
            spin.valueChanged.connect(lambda *_: self._emit())
            lay.addRow(f"[{i}]", spin)
            boxes.append(spin)

        def changed(*_):
            param.value = [b.value() for b in boxes]
            self._emit()

        for b in boxes:
            b.valueChanged.connect(changed)
        reset = QPushButton("↺")
        reset.setFixedWidth(28)
        reset.clicked.connect(lambda *_: self._reset_field(
            name, lambda v: setattr(self._target, name, v)))
        lay.addRow("", reset)
        self._form.addRow(name, row)
        self._fields[name] = boxes

    def _add_matrix_row(self, name: str, param: Matrix3Parameter) -> None:
        """Editable 3x3 grid (inertia tensors were read-only in M5)."""
        row = QWidget()
        lay = QFormLayout(row)
        lay.addRow("unit", QLabel(param.unit))
        grid = QGridLayout()
        boxes = []
        for r in range(3):
            for c in range(3):
                spin = QDoubleSpinBox()
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(12)
                spin.setValue(param.value[r][c])
                grid.addWidget(spin, r, c)
                boxes.append((r, c, spin))
        lay.addRow(grid)

        def changed(*_):
            param.value = [[next(s for rr, cc, s in boxes if (rr, cc) == (r, c)).value()
                            for c in range(3)] for r in range(3)]
            self._emit()

        for _, _, b in boxes:
            b.valueChanged.connect(changed)
        reset = QPushButton("↺")
        reset.setFixedWidth(28)
        reset.clicked.connect(lambda *_: self._reset_field(
            name, lambda v: setattr(self._target, name, v)))
        lay.addRow("", reset)
        self._form.addRow(name, row)
        self._fields[name] = [b for _, _, b in boxes]

    def _add_plain_row(self, name: str, value) -> None:
        edit = QLineEdit(str(value))
        edit.editingFinished.connect(self._emit)
        self._form.addRow(name, edit)
        self._fields[name] = [edit]
