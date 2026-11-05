"""Time-series scope (M6 Task 3; supersedes the M5 single-channel view).

Simulation-Data-Inspector style: channel tree with checkboxes (overlays),
a draggable cursor linked across the app (``cursor_moved`` drives the 3D
panel's frame seek), and a phase-average overlay toggle for periodic records.

The widget stores exactly the arrays it displays (self.plotted, per-channel
buffers) so the demo-plot freshness test can compare against freshly
recomputed data (brief §10 M5).
"""

from __future__ import annotations

try:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


class TimeSeriesPanel(QWidget if _HAVE_QT else object):
    """Channel tree + overlay scope + linked cursor."""

    cursor_moved = Signal(float) if _HAVE_QT else None

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 + pyqtgraph required for the time-series panel")
        super().__init__(parent)
        self._ts = None
        self._overlay: tuple | None = None  # (TimeSeries, label) for compare (M6 T5)
        self.plotted: tuple = ()  # (t, y, channel_name) of the last replotted channel
        self._items: dict[str, QTreeWidgetItem] = {}
        self._suppress = False

        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.tree)
        self.avg_toggle = QCheckBox("phase-average overlay")
        self.avg_toggle.setEnabled(False)
        self.avg_toggle.toggled.connect(lambda _c: self._replot_all())
        left.addWidget(self.avg_toggle)
        layout.addLayout(left, stretch=0)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(4, 4))
        self.cursor = pg.InfiniteLine(angle=90, movable=True,
                                      pen=pg.mkPen("#E0B040", width=1))
        self.cursor.setVisible(False)
        self.cursor.sigPositionChanged.connect(self._cursor_changed)
        self.plot.addItem(self.cursor)
        layout.addWidget(self.plot, stretch=1)

    # --- record loading --------------------------------------------------------

    def set_record(self, ts) -> None:
        self._ts = ts
        self._suppress = True
        self.tree.clear()
        self._items = {}
        groups: dict[str, QTreeWidgetItem] = {}
        for name in ts.channel_names():
            group_name, _, leaf = name.partition("/")
            if group_name not in groups:
                groups[group_name] = QTreeWidgetItem(self.tree, [group_name])
                groups[group_name].setExpanded(True)
            parent = groups[group_name] if leaf else self.tree
            item = QTreeWidgetItem(parent, [name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            self._items[name] = item
        period = ts.meta.get("wingbeat_period_s") if ts.meta else None
        self.avg_toggle.setEnabled(period is not None)
        self._period = period
        self._suppress = False
        # default: check the first channel so the scope is never blank
        names = ts.channel_names()
        if names:
            self._items[names[0]].setCheckState(0, Qt.CheckState.Checked)
            self._replot(names[0])

    def set_overlay(self, ts, label: str) -> None:
        """Overlay another record's channels (dashed) on the current one."""
        self._overlay = (ts, label)
        self._replot_all()

    def clear_overlay(self) -> None:
        self._overlay = None
        self._replot_all()

    # --- plotting ---------------------------------------------------------------

    def _checked(self) -> list[str]:
        return [n for n, it in self._items.items()
                if it.checkState(0) == Qt.CheckState.Checked]

    def _on_item_changed(self, _item, _col):
        if not self._suppress:
            self._replot_all()

    def _replot(self, channel: str) -> None:
        """Plot one channel (kept API for the freshness test + default view)."""
        self._replot_all(focus=channel)

    def _replot_all(self, focus: str | None = None) -> None:
        if self._ts is None:
            return
        channels = self._checked() or ([focus] if focus else [])
        self.plot.clear()
        self.plot.addItem(self.cursor)
        for i, name in enumerate(channels):
            y = self._ts[name]
            color = pg.intColor(i)
            if y.ndim > 1:
                for k in range(y.shape[1]):
                    self.plot.plot(self._ts.t, y[:, k], pen=pg.intColor(i + k),
                                   name=f"{name}[{k}]")
            else:
                self.plot.plot(self._ts.t, y, pen=color, name=name)
            if self.avg_toggle.isChecked() and self._period and y.ndim == 1:
                from fwmav_sim.analysis.averaging import phase_average

                ph, mean, _spread = phase_average(self._ts, name, self._period)
                self.plot.plot(ph * self._period, mean,
                               pen=pg.mkPen(color, style=Qt.PenStyle.DashLine),
                               name=f"{name} (avg)")
            self.plotted = (self._ts.t.copy(), y.copy(), name)
            # compare overlay: same channel from the second record, dashed
            if self._overlay is not None:
                ots, olabel = self._overlay
                if name in ots.channel_names() and ots[name].shape == y.shape:
                    oy = ots[name]
                    if oy.ndim > 1:
                        for k in range(oy.shape[1]):
                            self.plot.plot(ots.t, oy[:, k],
                                           pen=pg.mkPen(pg.intColor(i + k),
                                                        style=Qt.PenStyle.DashLine),
                                           name=f"{olabel}:{name}[{k}]")
                    else:
                        self.plot.plot(ots.t, oy,
                                       pen=pg.mkPen(color, style=Qt.PenStyle.DashLine),
                                       name=f"{olabel}:{name}")
        self.plot.setLabel("bottom", "t [s]")
        if focus:
            self.plotted = (self._ts.t.copy(), self._ts[focus].copy(), focus)

    # --- cursor linkage ------------------------------------------------------------

    def _cursor_changed(self):
        t = float(self.cursor.value())
        self.cursor_moved.emit(t)

    def set_cursor(self, t: float) -> None:
        self.cursor.setVisible(True)
        self.cursor.setValue(t)
