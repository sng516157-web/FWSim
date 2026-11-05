"""Vehicle tree inspector (M5 Task 2): bodies, joints, wings with
provenance badges (estimated values visually distinct, brief §14.6).
"""

from __future__ import annotations

try:
    from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
    from PySide6.QtWidgets import QTreeView, QVBoxLayout, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

from fwmav_sim.vehicle.schema import VehicleDefinition

_BADGE_COLORS = {
    "measured": "#1a7a3c",
    "literature": "#1a5f9e",
    "estimated": "#b0660a",
    "fitted": "#7a1a9e",
}


def _badge_item(prov) -> QStandardItem:
    item = QStandardItem(f"[{prov.source}]")
    item.setForeground(QColor(_BADGE_COLORS.get(prov.source, "#666666")))
    return item


class InspectorPanel(QWidget if _HAVE_QT else object):
    """Tree view of a VehicleDefinition; selection signals the parameter editor."""

    def __init__(self, on_select, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the inspector panel")
        super().__init__(parent)
        self._on_select = on_select
        self.tree = QTreeView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["object", "provenance"])
        self.tree.setModel(self.model)
        self.tree.setColumnWidth(0, 260)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.tree.selectionModel().currentChanged.connect(self._selected)
        self._vehicle: VehicleDefinition | None = None
        self._index: dict = {}

    def set_vehicle(self, vehicle: VehicleDefinition) -> None:
        self._vehicle = vehicle
        self._index.clear()
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["object", "provenance"])
        root = self.model.invisibleRootItem()
        root.appendRow([QStandardItem(f"vehicle: {vehicle.name}"), QStandardItem("")])

        bodies = QStandardItem("bodies")
        for b in vehicle.bodies:
            node = QStandardItem(f"{b.name}  (mass {b.mass.value} {b.mass.unit})")
            bodies.appendRow([node, _badge_item(b.mass.provenance)])
            self._index[id(node)] = ("body", b.name)
        root.appendRow([bodies, QStandardItem("")])

        joints = QStandardItem("joints")
        for j in vehicle.joints:
            node = QStandardItem(f"{j.name}: {j.parent_body} -> {j.child_body} ({j.type})")
            joints.appendRow([node, QStandardItem("")])
            self._index[id(node)] = ("joint", j.name)
        root.appendRow([joints, QStandardItem("")])

        wings = QStandardItem("wings")
        for w in vehicle.wings:
            node = QStandardItem(f"{w.name}  ({w.side}, span {w.span.value} {w.span.unit})")
            wings.appendRow([node, _badge_item(w.span.provenance)])
            self._index[id(node)] = ("wing", w.name)
        root.appendRow([wings, QStandardItem("")])

        aero = QStandardItem(f"aero: {vehicle.aero.name}")
        root.appendRow([aero, QStandardItem("")])
        self._index[id(aero)] = ("aero", None)
        self.tree.expandAll()

    def set_experiment(self, exp) -> None:
        """Add an experiment section: kinematics (and per-wing variants) become
        editable nodes alongside the vehicle tree (M6 Task 4)."""
        root = self.model.invisibleRootItem()
        kin = QStandardItem("experiment: kinematics")
        kmap = (exp.kinematics if isinstance(exp.kinematics, dict)
                else {"(all wings)": exp.kinematics})
        for wing_name, k in kmap.items():
            if k is None:
                continue
            node = QStandardItem(f"kinematics: {wing_name}")
            kin.appendRow([node, QStandardItem("")])
            self._index[id(node)] = ("kinematics", wing_name)
        root.appendRow([kin, QStandardItem("")])
        self.tree.expandAll()

    def _selected(self, current, _previous):
        node = self.model.itemFromIndex(current)
        key = self._index.get(id(node))
        if key and self._vehicle is not None:
            self._on_select(key)
