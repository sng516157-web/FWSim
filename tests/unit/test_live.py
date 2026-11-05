"""Live dynamics streaming tests (M6.75+): the session integrates the real
plant and streams frames — offline, with a mock MeshCat."""

from __future__ import annotations

import numpy as np
import pytest


class _MockNode:
    def __init__(self, rec):
        self._rec = rec

    def set_object(self, *_a, **_k):
        self._rec["set_object"] += 1

    def set_transform(self, tf):
        self._rec["set_transform"] += 1
        self._rec["last_tf"] = np.asarray(tf)

    def set_property(self, *_a):
        self._rec["set_property"] += 1


class _MockVis:
    """meshcat.Visualizer stand-in: path-indexed node recorder."""

    def __init__(self):
        self.rec = {"set_object": 0, "set_transform": 0, "set_property": 0}

    def __getitem__(self, path):
        return _MockNode(self.rec)

    def url(self):
        return "mock://live"


def test_live_session_advances_and_streams():
    from fwmav_sim.viz.live import LiveSession

    vis = _MockVis()
    session = LiveSession("experiments/m3_free_body_hover.yaml", vis)
    t0 = session.t
    y0 = session.y.copy()
    for _ in range(5):
        session.step_frame(0.005)
    assert session.t > t0
    assert not np.array_equal(session.y, y0)  # dynamics advanced
    assert vis.rec["set_transform"] > 5  # frames streamed


def test_live_session_records_full_channels(tmp_path):
    """A live session records every solver step; the saved record loads with
    the batch run-record schema."""
    from fwmav_sim.experiment.record import load_record
    from fwmav_sim.viz.live import LiveSession

    vis = _MockVis()
    session = LiveSession("experiments/m3_free_body_hover.yaml", vis,
                          out_root=tmp_path)
    for _ in range(3):
        session.step_frame(0.003)
    assert session._recorder.n_samples > 0
    rec_dir = session.save_record()
    ts, meta = load_record(rec_dir)
    assert meta["time_series_meta"]["session"] == "live"
    for ch in ("p", "quat", "v_W", "omega_B", "reaction_torques",
               "wing_right/F_aero_total", "wing_right/F_trans",
               "wing_right/lever_arm", "com_W"):
        assert ch in ts.channel_names(), ch
    assert len(ts.t) == session._recorder.n_samples


def test_live_rejects_non_free_body():
    from fwmav_sim.viz.live import LiveSession

    with pytest.raises(ValueError, match="free_body"):
        LiveSession("experiments/m1_fixed_body_hover.yaml", _MockVis())
