"""Platform tree editor canvas (M8 Tasks 2-3).

The platform is a TREE (bodies -> joints -> wings), not a free graph —
this is a tree view with typed add/remove operations, backed by
vehicle/platform_builder.PlatformBuilder. URDF import is a node source.
Compiles to vehicle YAML (verbose provenance form).

Graph ops are Qt-free (on PlatformTree); the QTreeWidget is a thin view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fwmav_sim.vehicle.platform_builder import PlatformBuilder

try:
    from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
    _HAVE_QT = True
except ImportError:  # pragma: no cover
    _HAVE_QT = False


class PlatformTree:
    """Qt-free model: a PlatformBuilder plus tree-integrity rules."""

    def __init__(self, name: str = "new_platform"):
        self.builder = PlatformBuilder(name)

    # node sources -----------------------------------------------------

    def import_urdf(self, path: str | Path) -> None:
        other = PlatformBuilder.from_urdf(path)
        self.builder.bodies.extend(other.bodies)
        self.builder.joints.extend(other.joints)

    # edits --------------------------------------------------------------

    def add_body(self, *a, **kw):
        self.builder.add_body(*a, **kw)

    def add_joint(self, *a, **kw):
        self.builder.add_joint(*a, **kw)

    def add_wing(self, *a, **kw):
        self.builder.add_wing(*a, **kw)

    def remove(self, kind: str, name: str) -> None:
        if kind == "wing":
            self.builder.remove_wing(name)
        elif kind == "body":
            if any(j["parent_body"] == name or j["child_body"] == name
                   for j in self.builder.joints):
                raise ValueError(f"body {name!r} still has joints — remove them first")
            self.builder.bodies = [b for b in self.builder.bodies if b["name"] != name]
        elif kind == "joint":
            self.builder.joints = [j for j in self.builder.joints if j["name"] != name]
        else:
            raise ValueError(f"unknown node kind {kind!r}")

    # compile ------------------------------------------------------------

    def compile(self):
        return self.builder.to_vehicle()

    def save(self, path: str | Path) -> Path:
        return self.builder.save_vehicle(path)


if _HAVE_QT:

    class PlatformTreeWidget(QTreeWidget):
        """Tree view over a PlatformTree; edits go through the model."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.model_tree = PlatformTree()
            self.setHeaderLabels(["node", "detail"])
            self.refresh()

        def refresh(self):
            self.clear()
            b = self.model_tree.builder
            root = QTreeWidgetItem([b.name, "platform"])
            self.addTopLevelItem(root)
            bodies = QTreeWidgetItem(["bodies", ""])
            root.addChild(bodies)
            for bd in b.bodies:
                bodies.addChild(QTreeWidgetItem(
                    [bd["name"], f"mass {bd['mass']['value']} kg"]))
            joints = QTreeWidgetItem(["joints", ""])
            root.addChild(joints)
            for j in b.joints:
                joints.addChild(QTreeWidgetItem(
                    [j["name"], f"{j['type']}: {j['parent_body']} -> {j['child_body']}"]))
            wings = QTreeWidgetItem(["wings", ""])
            root.addChild(wings)
            for w in b.wings:
                wings.addChild(QTreeWidgetItem(
                    [w["name"], f"{w['side']}, span {w['span']['value']} m"]))
            root.setExpanded(True)
