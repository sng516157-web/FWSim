"""Wing planform editor tests (M9 T1, offscreen)."""

from __future__ import annotations

import os

import pytest

from fwmav_sim.ui.nodes.planform_canvas import PlanformOutline

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_default_outline_is_rectangular():
    o = PlanformOutline()
    st = o.stations()
    assert len(st) == 2
    assert st[0] == {"y": 0.0, "chord": 0.2, "le_offset": 0.05}
    assert o.area == pytest.approx(0.2 * 0.3)
    assert o.mac == pytest.approx(0.2)
    assert o.span == 0.3


def test_stations_roundtrip():
    st = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
          {"y": 0.15, "chord": 0.18, "le_offset": 0.06},
          {"y": 0.3, "chord": 0.05, "le_offset": 0.12}]
    o = PlanformOutline(st)
    assert o.stations() == st


def test_move_point_clamps_to_neighbours():
    o = PlanformOutline()
    o.move_point("le", 0, 0.08, 0.5)  # root station: y clamped to 0
    assert o.stations()[0]["y"] == 0.0
    assert o.stations()[0]["le_offset"] == 0.08


def test_move_point_reorders_nothing():
    o = PlanformOutline([{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                         {"y": 0.1, "chord": 0.2, "le_offset": 0.05},
                         {"y": 0.3, "chord": 0.2, "le_offset": 0.05}])
    o.move_point("te", 1, -0.15, 0.9)  # y clamped below next station (0.3)
    ys = [s["y"] for s in o.stations()]
    assert ys[1] < 0.3


def test_insert_and_remove_station():
    o = PlanformOutline()
    o.insert_station(0.15)
    st = o.stations()
    assert len(st) == 3
    assert st[1]["y"] == pytest.approx(0.15)
    assert st[1]["chord"] == pytest.approx(0.2)
    o.remove_station(1)
    assert len(o.stations()) == 2


def test_remove_end_stations_rejected():
    o = PlanformOutline()
    with pytest.raises(ValueError, match="root and tip"):
        o.remove_station(0)
    with pytest.raises(ValueError, match="root and tip"):
        o.remove_station(1)


def test_zero_chord_rejected():
    with pytest.raises(ValueError, match="chord"):
        PlanformOutline([{"y": 0.0, "chord": 0.0, "le_offset": 0.05},
                         {"y": 0.3, "chord": 0.2, "le_offset": 0.05}])


def test_mac_trapezoid():
    # trapezoid c: 0.2 -> 0.1 over 0.3 m; MAC = (2/3)(cr+ct - cr*ct/(cr+ct))
    o = PlanformOutline([{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                         {"y": 0.3, "chord": 0.1, "le_offset": 0.08}])
    cr, ct = 0.2, 0.1
    mac = (2.0 / 3.0) * (cr + ct - cr * ct / (cr + ct))
    assert o.mac == pytest.approx(mac, rel=1e-9)


def test_qt_scene_drag_cycle():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from fwmav_sim.ui.nodes.planform_canvas import PlanformScene

    app = QApplication.instance() or QApplication([])
    scene = PlanformScene()
    assert scene._poly is not None
    n0 = scene.outline.n_stations
    scene.outline.insert_station(0.15)
    scene.redraw()
    assert scene.outline.n_stations == n0 + 1
    scene.apply_drags()  # points write back cleanly
    assert scene.outline.n_stations == n0 + 1
