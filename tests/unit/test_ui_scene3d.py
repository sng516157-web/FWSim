"""3D panel playback tests (post-M5): frame stepping, slider sync, play/pause.

Offscreen Qt; the MeshCat builders/renderers are monkeypatched so the test
covers the GUI logic without a server (server rendering is the M5 human-check
item; build/render paths are covered by viz tests).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Ctx:
    class ts:
        t = np.linspace(0.0, 2.0, 50)


def _fake_record(tmp_path, kind):
    d = tmp_path / "rec"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({"time_series_meta": {"kind": kind}}), encoding="utf-8")
    (d / "config.yaml").write_text("# placeholder; builders are monkeypatched\n",
                                   encoding="utf-8")
    return d


def _patch_scene(monkeypatch, panel, kind):
    import fwmav_sim.viz.meshcat_scene as ms

    panel._vis = object()  # skip the server
    monkeypatch.setattr(panel, "start_server", lambda: "mock://url")
    calls = []
    if kind == "free_body":
        monkeypatch.setattr(ms, "build_free_body_replay",
                            lambda vis, rd: (_Ctx(), 50))
        monkeypatch.setattr(ms, "render_free_body_frame",
                            lambda vis, ctx, i: calls.append(i))
    else:
        monkeypatch.setattr(ms, "build_kinematics_replay",
                            lambda vis, cfg, vehicle_path=None: ({"k": 1}, 0.1, None, None))
        monkeypatch.setattr(ms, "pose_frame",
                            lambda vis, state, t, model, env: calls.append(t))
    # scene3d imports builders/renderers from meshcat_scene at call time
    return calls


def test_record_config_vehicle_path_resolution(tmp_path, monkeypatch):
    """Regression: a config.yaml inside a run record has its vehicle path
    relative to the ORIGINAL experiment location — the record's own
    vehicle.yaml copy must be used instead (empty-scene bug)."""
    import fwmav_sim.viz.meshcat_scene as ms

    record = tmp_path / "outputs" / "rec"
    record.mkdir(parents=True)
    (record / "config.yaml").write_text(
        "name: t\nkind: fixed_body\nvehicle: ../reference_vehicles/bio_abdomen_fwmav.yaml\n",
        encoding="utf-8")
    (record / "vehicle.yaml").write_text("name: record_copy\n", encoding="utf-8")
    seen = {}
    from types import SimpleNamespace

    def fake_load_vehicle(p):
        seen["path"] = Path(p)
        return SimpleNamespace(wings=[], aero=None,
                               blade_elements=SimpleNamespace(n_elements=4))

    monkeypatch.setattr(ms, "load_vehicle", fake_load_vehicle)
    fake_kin = SimpleNamespace(flap_frequency=SimpleNamespace(value=10.0))
    monkeypatch.setattr(ms, "load_experiment",
                        lambda p: SimpleNamespace(vehicle="../reference_vehicles/x.yaml",
                                                  kinematics=fake_kin, environment=None))
    monkeypatch.setattr(ms, "environment_from_spec", lambda spec: None)
    monkeypatch.setattr(ms, "QuasiSteadyBladeElement", lambda aero: None)
    monkeypatch.setattr(ms, "build_scene", lambda *a, **k: {"wings": []})
    ms.build_kinematics_replay(object(), record / "config.yaml",
                               vehicle_path=record / "vehicle.yaml")
    assert seen["path"] == record / "vehicle.yaml"


def test_free_body_playback_steps_and_syncs(qapp, tmp_path, monkeypatch):
    from fwmav_sim.ui.scene3d import Scene3DPanel

    panel = Scene3DPanel()
    calls = _patch_scene(monkeypatch, panel, "free_body")
    panel.set_record(_fake_record(tmp_path, "free_body"))
    assert calls == [0]  # frame 0 shown paused
    assert panel.frame_slider.maximum() == 49
    assert not panel._playing

    panel.speed_combo.setCurrentText("1.0")
    panel.toggle_play()
    assert panel._playing
    for _ in range(4):
        panel._tick()  # 4 x 30 ms at 1.0x over a 2 s / 50-frame record
    # advance per tick = 0.03 * 1.0 * 50 / 2.0 = 0.75 frames
    assert panel._frame == pytest.approx(3.0)
    assert calls[-1] == int(panel._frame)
    assert panel.frame_slider.value() == calls[-1]
    assert panel.frame_label.text() == f"{calls[-1] + 1}/50"

    panel.toggle_play()
    assert not panel._playing
    n_calls = len(calls)
    panel._timer.stop()
    assert len(calls) == n_calls  # no further rendering while paused


def test_kinematics_playback_wraps(qapp, tmp_path, monkeypatch):
    from fwmav_sim.ui.scene3d import Scene3DPanel

    panel = Scene3DPanel()
    calls = _patch_scene(monkeypatch, panel, "fixed_body")
    panel.set_record(_fake_record(tmp_path, "fixed_body"))
    assert calls == [0.0]
    panel.speed_combo.setCurrentText("1.0")
    panel.toggle_play()
    # tick enough to wrap past one period (120 frames, period 0.1 s)
    for _ in range(200):
        panel._tick()
    assert 0.0 <= panel._frame < 120
    assert len(calls) == 201


def test_set_experiment_previews_kinematics(qapp, tmp_path, monkeypatch):
    """Opening an experiment poses the vehicle/wings in 3D (paused frame 0)."""
    from fwmav_sim.ui.scene3d import Scene3DPanel

    panel = Scene3DPanel()
    calls = _patch_scene(monkeypatch, panel, "fixed_body")
    panel.set_experiment(tmp_path / "exp.yaml")  # builder monkeypatched
    assert calls == [0.0]
    assert panel.play_button.isEnabled()
    assert panel.frame_slider.maximum() == 119
    panel.toggle_play()
    for _ in range(10):
        panel._tick()
    assert len(calls) == 11


def test_slider_seek(qapp, tmp_path, monkeypatch):
    from fwmav_sim.ui.scene3d import Scene3DPanel

    panel = Scene3DPanel()
    calls = _patch_scene(monkeypatch, panel, "free_body")
    panel.set_record(_fake_record(tmp_path, "free_body"))
    panel.frame_slider.setValue(17)
    panel._slider_released()
    assert calls[-1] == 17
    assert panel.frame_label.text() == "18/50"
