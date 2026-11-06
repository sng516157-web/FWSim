"""Qt-native OpenGL 3D scene (M9 T5 — replaces MeshCat in the GUI).

Why this exists: the user chose a native Qt 3D view over MeshCat (browser
tab + localhost server + fullscreen glitches). Qt3D's Python bindings were
dropped after Qt6 (PySide6 6.12 ships an empty Qt3DExtras), and QtQuick3D
is QML-only — so this is a compact QOpenGLWidget renderer we own:
orbit camera, grid, wing meshes (custom planforms honoured), force arrows,
frame updates for replay scrubbing and live streaming.

Physics never enters here: the widget consumes world-frame geometry per
frame, exactly like the MeshCat layer it replaces (NED mapped at the view
boundary, same NED_TO_VIEW as meshcat_scene).
"""

from __future__ import annotations

import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMatrix4x4, QVector3D, QWheelEvent
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
    _HAVE_GL = True
except ImportError:  # pragma: no cover
    _HAVE_GL = False

from fwmav_sim.viz.meshcat_scene import NED_TO_VIEW  # noqa: F401  (re-exported convention)


_VERT_SRC = """
#version 120
attribute vec3 pos;
attribute vec3 color;
uniform mat4 mvp;
varying vec3 v_color;
void main() { gl_Position = mvp * vec4(pos, 1.0); v_color = color; }
"""

_FRAG_SRC = """
#version 120
varying vec3 v_color;
void main() { gl_FragColor = vec4(v_color, 1.0); }
"""


if _HAVE_GL:

    class GLSceneWidget(QOpenGLWidget):
        """Orbit-camera 3D scene: grid + wing meshes + force arrows."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._yaw, self._pitch, self._dist = 0.6, 0.4, 1.2
            self._last_pos = None
            self._prog: QOpenGLShaderProgram | None = None
            # scene buffers: (mode, vertex ndarray [x,y,z,r,g,b]*N, count)
            self._lines: list[np.ndarray] = []   # GL_LINES batches
            self._tris: list[np.ndarray] = []    # GL_TRIANGLES batches
            self._grid = self._make_grid()

        # --- static content ------------------------------------------------

        @staticmethod
        def _make_grid(extent: float = 1.0, step: float = 0.1) -> np.ndarray:
            pts = []
            n = int(2 * extent / step) + 1
            for i in range(n):
                c = -extent + i * step
                major = abs(c) < 1e-9
                col = (0.45, 0.45, 0.5) if major else (0.3, 0.3, 0.33)
                pts += [[c, 0, -extent, *col], [c, 0, extent, *col]]
                pts += [[-extent, 0, c, *col], [extent, 0, c, *col]]
            return np.asarray(pts, dtype=np.float32)

        def set_wing_meshes(self, meshes: list[tuple[np.ndarray, np.ndarray]]) -> None:
            """meshes: (vertices (n,3) WR, faces (m,3)) per wing — already in
            VIEW coords (apply NED_TO_VIEW upstream)."""
            self._tris = []
            for verts, faces in meshes:
                col = np.tile(np.array([0.27, 0.67, 1.0], dtype=np.float32),
                              (len(verts), 1))
                data = np.concatenate([verts, col], axis=1).astype(np.float32)
                idx = faces.astype(np.uint32).ravel()
                self._tris.append(data[idx])  # flattened triangles

        def set_force_arrows(self, arrows: list[tuple[np.ndarray, np.ndarray]]) -> None:
            """arrows: (origin (3,), vector (3,)) in VIEW coords."""
            self._lines = []
            for o, v in arrows:
                tip = o + v
                seg = np.array([[*o, 1, 0.2, 0.2], [*tip, 1, 0.2, 0.2]],
                               dtype=np.float32)
                self._lines.append(seg)

        # --- GL lifecycle -----------------------------------------------------

        def initializeGL(self):
            from OpenGL import GL

            GL.glClearColor(0.13, 0.14, 0.16, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            self._prog = QOpenGLShaderProgram()
            self._prog.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERT_SRC)
            self._prog.addShaderFromSourceCode(QOpenGLShader.Fragment, _FRAG_SRC)
            self._prog.link()

        def paintGL(self):
            from OpenGL import GL

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            if self._prog is None:
                return
            self._prog.bind()
            mvp = self._mvp()
            self._prog.setUniformValue("mvp", mvp)
            self._draw(GL.GL_LINES, self._grid)
            for batch in self._lines:
                GL.glLineWidth(2.0)
                self._draw(GL.GL_LINES, batch)
            GL.glLineWidth(1.0)
            for batch in self._tris:
                self._draw(GL.GL_TRIANGLES, batch)
            self._prog.release()

        def _draw(self, mode, data: np.ndarray):
            from OpenGL import GL

            data = np.ascontiguousarray(data, dtype=np.float32)
            stride = 6 * 4
            ptr = data.ctypes.data
            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride, ptr)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride, ptr + 12)
            GL.glDrawArrays(mode, 0, len(data))
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(1)

        def _mvp(self) -> QMatrix4x4:
            proj = QMatrix4x4()
            proj.perspective(45.0, max(self.width(), 1) / max(self.height(), 1),
                             0.01, 100.0)
            view = QMatrix4x4()
            eye = QVector3D(
                self._dist * np.cos(self._pitch) * np.cos(self._yaw),
                self._dist * np.sin(self._pitch),
                self._dist * np.cos(self._pitch) * np.sin(self._yaw))
            view.lookAt(eye, QVector3D(0, 0, 0.15), QVector3D(0, 1, 0))
            return proj * view

        # --- camera interaction ------------------------------------------------

        def mousePressEvent(self, ev):
            self._last_pos = ev.position()

        def mouseMoveEvent(self, ev):
            if self._last_pos is None:
                return
            d = ev.position() - self._last_pos
            self._yaw += d.x() * 0.01
            self._pitch = float(np.clip(self._pitch + d.y() * 0.01, -1.4, 1.4))
            self._last_pos = ev.position()
            self.update()

        def mouseReleaseEvent(self, ev):
            self._last_pos = None

        def wheelEvent(self, ev: QWheelEvent):
            self._dist = float(np.clip(self._dist * (0.9 if ev.angleDelta().y() > 0
                                                     else 1.1), 0.1, 10.0))
            self.update()
