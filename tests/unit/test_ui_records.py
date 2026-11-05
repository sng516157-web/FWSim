"""Records browser tests (M6 Task 5): discovery, open, compare overlay."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication

from fwmav_sim.core.timeseries import TimeSeries


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _record(root, name, kind="fixed_body", status="OK", mtime_shift=0):
    d = root / name
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({
        "time_series_meta": {"kind": kind}, "status": status}), encoding="utf-8")
    return d


def test_scan_records_newest_first_and_status(tmp_path):
    from fwmav_sim.ui.records import scan_records

    d_old = _record(tmp_path, "a_old")
    d_new = _record(tmp_path, "b_new", kind="free_body", status="FAILED")
    import os
    import time

    now = time.time()
    for d, mtime in ((d_old, now - 100), (d_new, now)):
        os.utime(d / "metadata.json", (mtime, mtime))  # deterministic ordering
    recs = scan_records(tmp_path)
    assert [r["name"] for r in recs] == ["b_new", "a_old"]
    assert recs[0]["status"] == "FAILED" and recs[0]["kind"] == "free_body"


def test_browser_open_and_compare_callbacks(qapp, tmp_path):
    from fwmav_sim.ui.records import RecordsPanel

    _record(tmp_path, "r1")
    _record(tmp_path, "r2", kind="free_body")
    opened, compared = [], []
    panel = RecordsPanel(tmp_path, opened.append, compared.append)
    assert panel.list.count() == 2

    item = panel.list.item(0)
    panel._open_item(item)
    assert opened

    # context menu compare path (call the handler logic directly)
    panel.list.setCurrentRow(1)
    panel._on_compare(panel._records[1]["dir"])
    assert compared


def test_scope_overlay(qapp):
    from fwmav_sim.ui.timeseries import TimeSeriesPanel

    t = np.linspace(0, 1, 50)
    ts1 = TimeSeries(t=t, channels={"w/F": np.sin(t)}, meta={})
    ts2 = TimeSeries(t=t, channels={"w/F": np.cos(t)}, meta={})
    panel = TimeSeriesPanel()
    panel.set_record(ts1)
    panel.set_overlay(ts2, "other")
    items = panel.plot.listDataItems()
    assert len(items) == 2  # primary + dashed overlay
    panel.clear_overlay()
    assert len(panel.plot.listDataItems()) == 1
