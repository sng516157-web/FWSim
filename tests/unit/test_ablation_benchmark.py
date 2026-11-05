"""Ablation + benchmark unit tests (M6.75 Tasks 5-6, light paths only)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.analysis.ablation import _blocks, _metrics, format_ablation
from fwmav_sim.analysis.benchmark import format_table
from fwmav_sim.core.timeseries import TimeSeries


def _ts_fixed():
    t = np.arange(200) * 0.001
    return TimeSeries(t=t, channels={
        "wing_right/F_total": np.stack([np.ones_like(t), np.zeros_like(t),
                                        -np.ones_like(t)], axis=1),
    }, meta={"kind": "fixed_body", "wingbeat_period_s": 0.1})


def test_metrics_fixed_body():
    m = _metrics(_ts_fixed())
    assert m["wing_right.meanFx"] == pytest.approx(1.0)
    assert m["wing_right.meanFz"] == pytest.approx(-1.0)


def test_blocks_detection():
    veh = {"aero": {"unsteady": {"wake_capture": {}}, "name": "m"},
           "wings": [{"name": "w", "mass": 0.001}], "abdomen_aero": None}
    exp = {"isolation": {}}
    labels = [label for label, _ in _blocks(exp, veh)]
    assert "unsteady_terms_off" in labels
    assert "wing_inertia_off" in labels
    veh2 = {"aero": {"name": "m"}, "wings": [{"name": "w"}]}
    assert _blocks(exp, veh2) == []


def test_format_ablation_table():
    base = {"wing_right.meanFz": -1.0}
    rows = [{"block": "unsteady_terms_off",
             "metrics": {"wing_right.meanFz": -0.9},
             "delta": {"wing_right.meanFz": 0.1}}]
    md = format_ablation("test_exp", base, rows)
    assert "baseline (all blocks on)" in md
    assert "unsteady_terms_off" in md
    assert "-0.9" in md


def test_format_benchmark_table():
    rows = [{"experiment": "e1", "kind": "fixed_body", "sim_time_s": 0.333,
             "wall_time_s": 10.0, "sim_x_realtime": 0.0333,
             "blocks": ["fixed_body"]}]
    table = format_table(rows)
    assert "e1" in table and "x realtime" in table
