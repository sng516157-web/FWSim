"""3D view (M5 Task 4): MeshCat in QWebEngineView, driven by a meshcat server
in the same process (DR-002). Content comes from the vehicle definition and
run-record channels — CoM, lever arms, force overlays — never hard-coded.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAVE_WEBENGINE = True
except ImportError:
    _HAVE_WEBENGINE = False


class Scene3DPanel(QWidget if _HAVE_QT else object):
    """MeshCat browser view + scene content management."""

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the 3D panel")
        super().__init__(parent)
        self._vis = None
        self._url: str | None = None
        layout = QVBoxLayout(self)
        if _HAVE_WEBENGINE:
            self._web = QWebEngineView()
            layout.addWidget(self._web)
        else:
            from PySide6.QtWidgets import QLabel

            layout.addWidget(QLabel("QtWebEngine unavailable; open the MeshCat "
                                    "URL in a browser (printed on run)."))
            self._web = None

    @property
    def url(self) -> str | None:
        return self._url

    def start_server(self) -> str:
        """Start the shared MeshCat server and show it."""
        import meshcat

        self._vis = meshcat.Visualizer()
        self._url = self._vis.url()
        if self._web is not None:
            self._web.setUrl(QUrl(self._url))
        return self._url

    def show_record_frame(self, record_dir, frame: int = 0) -> None:
        """Render one frame of a free-body record: body pose, wings, CoM,
        lever arms, per-element force arrows (from the recorded state)."""
        if self._vis is None:
            self.start_server()
        from fwmav_sim.viz.meshcat_scene import render_record_frame

        render_record_frame(self._vis, record_dir, frame)
