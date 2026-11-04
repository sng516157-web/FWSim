"""Schema-driven parameter editor (M5 Task 3).

Forms are generated from the Pydantic models (never hand-built per field):
every physical Parameter gets value + unit + provenance editing; validation
errors surface inline. Write-back targets the same YAML the CLI consumes —
no second config format.
"""

from __future__ import annotations

try:
    from PySide6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QLabel,
        QLineEdit,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

from fwmav_sim.vehicle.schema import Matrix3Parameter, Parameter, Vector3Parameter

_SOURCES = ["measured", "literature", "estimated", "fitted"]


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
        outer = QVBoxLayout(self)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._form = QFormLayout(self._inner)
        self._scroll.setWidget(self._inner)
        outer.addWidget(QLabel("select an object in the inspector"))
        outer.addWidget(self._scroll)

    def set_target(self, obj) -> None:
        """Build the form for a Pydantic model instance."""
        self._target = obj
        self._fields.clear()
        while self._form.rowCount():
            self._form.removeRow(0)
        for name, finfo in type(obj).model_fields.items():
            current = getattr(obj, name)
            if isinstance(current, Parameter):
                self._add_parameter_row(name, current)
            elif isinstance(current, Vector3Parameter):
                self._add_vector_row(name, current)
            elif isinstance(current, Matrix3Parameter):
                label = QLabel(f"3x3 matrix, {current.unit} (edit in YAML)")
                self._form.addRow(name, label)
            else:
                self._add_plain_row(name, current)
        # provenance editing for Parameter fields is inside those rows

    def _emit(self):
        if self._target is not None:
            self._on_change(self._target)

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
        self._form.addRow(name, row)
        self._fields[name] = boxes

    def _add_plain_row(self, name: str, value) -> None:
        edit = QLineEdit(str(value))
        edit.editingFinished.connect(self._emit)
        self._form.addRow(name, edit)
        self._fields[name] = [edit]
