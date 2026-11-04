"""Time-series view (M5 Task 5): pyqtgraph plots of record channels, reusing
analysis/ products. The widget stores exactly the arrays it displays
(self.plotted) so the demo-plot freshness test can compare against freshly
recomputed data (brief §10 M5).
"""

from __future__ import annotations

try:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


class TimeSeriesPanel(QWidget if _HAVE_QT else object):
    """Channel selector + plot; data always from a loaded run record."""

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 + pyqtgraph required for the time-series panel")
        super().__init__(parent)
        self._ts = None
        self.plotted: tuple = ()  # (t, y, channel_name) currently displayed
        layout = QVBoxLayout(self)
        self.selector = QComboBox()
        self.selector.currentTextChanged.connect(self._replot)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.selector)
        layout.addWidget(self.plot)

    def set_record(self, ts) -> None:
        self._ts = ts
        self.selector.clear()
        self.selector.addItems(ts.channel_names())
        if ts.channel_names():
            self._replot(ts.channel_names()[0])

    def _replot(self, channel: str) -> None:
        if self._ts is None or not channel:
            return
        y = self._ts[channel]
        if y.ndim > 1:
            # vector channels: plot components with a legend-free overlay
            self.plot.clear()
            for k in range(y.shape[1]):
                self.plot.plot(self._ts.t, y[:, k], pen=pg.intColor(k))
            self.plotted = (self._ts.t.copy(), y.copy(), channel)
        else:
            self.plot.clear()
            self.plot.plot(self._ts.t, y)
            self.plotted = (self._ts.t.copy(), y.copy(), channel)
        self.plot.setLabel("left", channel)
