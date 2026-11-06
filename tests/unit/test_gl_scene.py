"""Qt-native GL scene tests (M9 T5, offscreen)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("OpenGL")

from PySide6.QtWidgets import QApplication  # noqa: E402

from fwmav_sim.viz.gl_scene import GLSceneWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_widget_constructs(app):
    w = GLSceneWidget()
    assert w._grid.shape[1] == 6  # x,y,z,r,g,b per vertex
    assert w._grid.shape[0] > 0


def test_set_meshes_and_arrows(app):
    w = GLSceneWidget()
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    w.set_wing_meshes([(verts, faces)])
    assert len(w._tris) == 1
    assert w._tris[0].shape == (3, 6)  # 3 verts x (pos+color)
    w.set_force_arrows([(np.zeros(3), np.array([0, 0, 0.5]))])
    assert len(w._lines) == 1
    assert w._lines[0].shape == (2, 6)


def test_camera_orbit_and_zoom(app):
    w = GLSceneWidget()
    yaw0, dist0 = w._yaw, w._dist
    w._yaw += 0.1
    w._dist *= 0.9
    assert w._yaw != yaw0 and w._dist < dist0
    mvp = w._mvp()
    assert not np.allclose(np.array(mvp.data()), 0.0)


def test_gl_init_offscreen(app):
    """paintGL must not raise when a GL context is available; skip cleanly
    when the offscreen platform can't create one."""
    w = GLSceneWidget()
    w.resize(320, 240)
    w.show()
    app.processEvents()
    if not w.isValid():
        pytest.skip("no GL context on the offscreen platform")
    w.grabFramebuffer()  # forces a real paint
    assert w._prog is not None
    w.close()
