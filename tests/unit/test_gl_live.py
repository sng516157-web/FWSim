"""GL live session tests (M9 T5, headless — stub GL widget)."""

from __future__ import annotations

import numpy as np
import pytest


class _StubGL:
    """Stands in for GLSceneWidget: records what it was given."""

    def __init__(self):
        self.meshes = None
        self.arrows = None
        self.updates = 0

    def set_wing_meshes(self, meshes):
        self.meshes = meshes

    def set_force_arrows(self, arrows):
        self.arrows = arrows

    def update(self):
        self.updates += 1


def test_gl_live_session_steps_and_records(tmp_path):
    from fwmav_sim.viz.gl_live import GLLiveSession

    gl = _StubGL()
    session = GLLiveSession("experiments/m3_free_body_hover.yaml", gl,
                            record=True, out_root=tmp_path)
    for _ in range(3):
        session.step_frame(0.02)  # 60 ms sim
    assert session.t > 0.05
    assert gl.meshes is not None and len(gl.meshes) == 2  # two wings
    verts, faces = gl.meshes[0]
    assert np.all(np.isfinite(verts))
    assert gl.updates == 3
    rec = session.save_record()
    assert rec is not None
    from fwmav_sim.experiment.record import load_record

    ts, meta = load_record(rec)
    assert len(ts.t) >= 2
    assert "p" in ts.channels


def test_gl_live_rejects_non_free_body(tmp_path):
    from fwmav_sim.viz.gl_live import GLLiveSession

    with pytest.raises(ValueError, match="free_body"):
        GLLiveSession("experiments/m1_fixed_body_hover.yaml", _StubGL(),
                      record=False, out_root=tmp_path)
