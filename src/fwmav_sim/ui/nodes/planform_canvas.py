"""Wing planform vector editor (M9 T1).

The user draws the half-wing outline in plan view: leading-edge and
trailing-edge polylines, dragged point by point (PlanformCreator2 style).
Every edit re-derives the stations (y, chord, le_offset), area, and MAC.

Qt-free model (`PlanformOutline`) holds the outline and enforces the
invariants (ordered stations, positive chord, root at y=0); the
QGraphicsScene canvas (`PlanformScene`) is a thin draggable view over it.
Edits round-trip to `planform_stations` YAML — the same schema field the
engine consumes (no second representation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OutlinePoint:
    """One editable outline point. kind: 'le' (leading edge) or 'te'."""

    kind: str  # 'le' | 'te'
    x: float  # x_WR (toward LE)
    y: float  # span station


class PlanformOutline:
    """Editable half-wing outline as LE/TE polylines sharing span stations.

    Invariants (enforced on every edit, loudly):
      - same number of LE and TE points, paired by span station
      - stations strictly increasing, first at y=0
      - chord = le.x - te.x > 0 at every station
    """

    def __init__(self, stations: list[dict] | None = None):
        """stations: [{'y':…, 'chord':…, 'le_offset':…}] (schema form)."""
        if stations is None:
            stations = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                        {"y": 0.3, "chord": 0.2, "le_offset": 0.05}]
        self._le = [(s["le_offset"], s["y"]) for s in stations]
        self._te = [(s["le_offset"] - s["chord"], s["y"]) for s in stations]
        self._check()

    def _check(self) -> None:
        if len(self._le) < 2:
            raise ValueError("outline needs at least 2 stations")
        ys = [p[1] for p in self._le]
        if ys[0] != 0.0:
            raise ValueError("first station must be at the root (y=0)")
        if any(b <= a for a, b in zip(ys, ys[1:])):
            raise ValueError("stations must be strictly increasing")
        for (lx, _), (tx, _) in zip(self._le, self._te):
            if lx - tx <= 0:
                raise ValueError("chord must be positive at every station")

    # --- reads ---------------------------------------------------------------

    @property
    def n_stations(self) -> int:
        return len(self._le)

    @property
    def span(self) -> float:
        return self._le[-1][1]

    def points(self) -> list[OutlinePoint]:
        return ([OutlinePoint("le", x, y) for x, y in self._le]
                + [OutlinePoint("te", x, y) for x, y in self._te])

    def stations(self) -> list[dict]:
        """Schema form (planform_stations)."""
        return [{"y": y, "chord": lx - tx, "le_offset": lx}
                for (lx, y), (tx, _) in zip(self._le, self._te)]

    @property
    def area(self) -> float:
        return float(sum(0.5 * (a["chord"] + b["chord"]) * (b["y"] - a["y"])
                         for a, b in zip(self.stations(), self.stations()[1:])))

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord: (1/A) ∫ c² dy."""
        num = den = 0.0
        for a, b in zip(self.stations(), self.stations()[1:]):
            dy = b["y"] - a["y"]
            num += dy * (a["chord"]**2 + a["chord"] * b["chord"] + b["chord"]**2) / 3.0
            den += 0.5 * (a["chord"] + b["chord"]) * dy
        return num / den if den > 0 else 0.0

    # --- edits ---------------------------------------------------------------

    def move_point(self, kind: str, index: int, x: float, y: float) -> None:
        """Drag one point. y is clamped between the neighbours (stations
        stay ordered); the root station stays at y=0."""
        lst = self._le if kind == "le" else self._te
        pts = [list(p) for p in lst]
        if index == 0:
            y = 0.0  # root station is pinned to the hinge line
        else:
            lo = pts[index - 1][1] + 1e-9
            hi = pts[index + 1][1] - 1e-9 if index < len(pts) - 1 else np.inf
            y = min(max(y, lo), hi)
        pts[index] = [x, y]
        if kind == "le":
            self._le = [(p[0], p[1]) for p in pts]
        else:
            self._te = [(p[0], p[1]) for p in pts]
        self._check()

    def insert_station(self, y: float) -> None:
        """Add a station at span y, linearly interpolated."""
        st = self.stations()
        ys = [s["y"] for s in st]
        if not 0.0 < y < ys[-1]:
            raise ValueError("insert inside the span")
        le = float(np.interp(y, ys, [s["le_offset"] for s in st]))
        ch = float(np.interp(y, ys, [s["chord"] for s in st]))
        i = max(k for k, yy in enumerate(ys) if yy < y)
        self._le.insert(i + 1, (le, y))
        self._te.insert(i + 1, (le - ch, y))
        self._check()

    def remove_station(self, index: int) -> None:
        if index in (0, len(self._le) - 1):
            raise ValueError("root and tip stations cannot be removed")
        del self._le[index]
        del self._te[index]
        self._check()


# ---------------------------------------------------------------------------
# Qt canvas
# ---------------------------------------------------------------------------

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
    from PySide6.QtWidgets import (
        QGraphicsEllipseItem,
        QGraphicsPolygonItem,
        QGraphicsScene,
        QGraphicsView,
    )
    _HAVE_QT = True
except ImportError:  # pragma: no cover
    _HAVE_QT = False

if _HAVE_QT:

    _R = 5.0

    class _DragPoint(QGraphicsEllipseItem):
        def __init__(self, scene_: "PlanformScene", kind: str, index: int):
            super().__init__(-_R, -_R, 2 * _R, 2 * _R)
            self.scene_, self.kind, self.index = scene_, kind, index
            self.setBrush(QBrush(QColor("#e0a030" if kind == "le" else "#30a0e0")))
            self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
            self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
            self.setZValue(2)

        def itemChange(self, change, value):
            if change == QGraphicsEllipseItem.ItemPositionChange and self.scene_:
                # canvas coords: x_pix = x * scale, y_pix = -y * scale (y up)
                s = self.scene_._scale
                return self.scene_.constrain_drag(self, value.x() / s,
                                                  -value.y() / s)
            return super().itemChange(change, value)

    class PlanformScene(QGraphicsScene):
        """Plan-view outline canvas over a PlanformOutline model."""

        def __init__(self, outline: PlanformOutline | None = None, parent=None):
            super().__init__(parent)
            self.outline = outline or PlanformOutline()
            self._scale = 800.0  # px per metre (set by fit on first draw)
            self._points: list[_DragPoint] = []
            self._poly: QGraphicsPolygonItem | None = None
            self.on_change = None  # callback after any edit
            self.redraw()

        def constrain_drag(self, item: "_DragPoint", x: float, y: float):
            """Clamp a drag to the model's invariants; returns canvas pos.
            x, y are model coords (m): x toward LE, y spanwise."""
            from PySide6.QtCore import QPointF

            pts = [p for p in self.outline.points() if p.kind == item.kind]
            i = item.index
            lo = 0.0 if i == 0 else pts[i - 1].y + 1e-4
            hi = pts[i + 1].y - 1e-4 if i < len(pts) - 1 else np.inf
            y_c = min(max(y, lo), hi)
            return QPointF(x * self._scale, -y_c * self._scale)

        def redraw(self):
            self.clear()
            self._points = []
            pts = self.outline.points()
            le = [(p.x, p.y) for p in pts if p.kind == "le"]
            te = [(p.x, p.y) for p in pts if p.kind == "te"]
            s = self._scale
            poly = QPolygonF()
            for x, y in le:
                poly.append(self._to_px(x, y))
            for x, y in reversed(te):
                poly.append(self._to_px(x, y))
            self._poly = QGraphicsPolygonItem(poly)
            self._poly.setBrush(QBrush(QColor(68, 170, 255, 120)))
            self._poly.setPen(QPen(QColor("#44AAFF"), 2))
            self.addItem(self._poly)
            for kind, seq in (("le", le), ("te", te)):
                for i, (x, y) in enumerate(seq):
                    d = _DragPoint(self, kind, i)
                    d.setPos(self._to_px(x, y))
                    self.addItem(d)
                    self._points.append(d)

        def _to_px(self, x, y):
            from PySide6.QtCore import QPointF

            return QPointF(x * self._scale, -y * self._scale)

        def apply_drags(self):
            """Write current point positions back into the model."""
            for d in self._points:
                pos = d.pos()
                self.outline.move_point(d.kind, d.index,
                                        pos.x() / self._scale,
                                        -pos.y() / self._scale)
            if self.on_change:
                self.on_change()

    class PlanformCanvas(QGraphicsView):
        def __init__(self, outline: PlanformOutline | None = None, parent=None):
            super().__init__(parent)
            self.setScene(PlanformScene(outline, self))
            self.setDragMode(QGraphicsView.NoDrag)

        def mouseReleaseEvent(self, ev):
            super().mouseReleaseEvent(ev)
            sc = self.scene()
            if isinstance(sc, PlanformScene):
                sc.apply_drags()
                sc.redraw()
