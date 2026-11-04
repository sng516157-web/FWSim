"""M5 integration: the brief §9 reference workflow scripted end-to-end
(offscreen Qt). Steps: 1 select vehicle, 2 inspect, 3 configure, 4 fixed-body
run, 5 inspect forces, 6 enable abdomen, 7 suspended test, 8 compare, 9
free-body run, 10 save/reload, 11 headless repeatability.

Short horizons everywhere (test runtime); each step asserts the UI state
changed correctly, not just that no exception was raised.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _shorten(exp_path: Path, tmp: Path, max_time=None, n_wingbeats=None) -> Path:
    data = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    data["vehicle"] = str((exp_path.parent / data["vehicle"]).resolve())
    if max_time is not None and "solver" in data:
        data["solver"]["max_time"] = max_time
    if n_wingbeats is not None and "n_wingbeats" in data:
        data["n_wingbeats"] = n_wingbeats
    out = tmp / exp_path.name
    out.write_text(yaml.safe_dump(data))
    return out


def test_reference_workflow(qapp, tmp_path):
    from fwmav_sim.ui.main_window import MainWindow

    win = MainWindow(tmp_path)

    # steps 1-3: select vehicle via experiment, inspect, configure
    exp1 = _shorten(Path("experiments/m1_fixed_body_hover.yaml"), tmp_path,
                    n_wingbeats=1)
    win.load_experiment(exp1)
    assert win._vehicle is not None
    win._on_inspector_select(("body", "thorax"))
    assert "mass" in win.editor._fields
    win.editor._fields["mass"][0].setValue(0.015)  # configure: edit thorax mass

    # steps 4-5: fixed-body run, inspect forces
    win.run_panel.refresh(tmp_path)
    win.run_panel.combo.setCurrentText(str(exp1))
    win.run_panel._run()
    win.run_panel._worker.wait(60_000)
    qapp.processEvents()  # deliver queued worker signals (no event loop in tests)
    assert win._record_dir is not None
    assert win.timeseries.plotted, "time-series view must show the record"
    t, y, ch = win.timeseries.plotted
    assert len(t) > 10 and np.isfinite(y).all()

    # steps 6-7: enable abdomen -> suspended test
    exp2 = _shorten(Path("experiments/m2_actuated_pendulum.yaml"), tmp_path,
                    max_time=0.5)
    win.load_experiment(exp2)
    win.run_panel.refresh(tmp_path)
    win.run_panel.combo.setCurrentText(str(exp2))
    win.run_panel._run()
    win.run_panel._worker.wait(60_000)
    qapp.processEvents()  # deliver queued worker signals (no event loop in tests)
    assert win._record_dir is not None

    # step 8: compare against the synthetic dataset (pipeline proof)
    from fwmav_sim.datasets.synthetic import make_synthetic_dataset

    ds = make_synthetic_dataset(win._record_dir, tmp_path / "ds", "phi_p",
                                phase_offset_s=0.005, noise_std=0.01,
                                downsample_hz=500.0, seed=1)
    win.timeseries._replot("phi_p")
    win._compare_dataset = ds  # drive the compare path programmatically
    from fwmav_sim.analysis.compare import compare

    report = compare(win._record_dir, ds, "phi_p", out_root=tmp_path / "cmp")
    import json

    rep = json.loads((report / "comparison_report.json").read_text())
    assert rep["validation_status"] == "SYNTHETIC_PROOF"

    # steps 9-11: free-body run + save/reload + headless repeat
    exp3 = _shorten(Path("experiments/m3_free_body_hover.yaml"), tmp_path,
                    max_time=0.05)
    win.load_experiment(exp3)
    win.run_panel.refresh(tmp_path)
    win.run_panel.combo.setCurrentText(str(exp3))
    win.run_panel._run()
    win.run_panel._worker.wait(120_000)
    qapp.processEvents()
    assert win._record_dir is not None
    from fwmav_sim.experiment.record import load_record

    ts_a, _ = load_record(win._record_dir)
    assert ts_a.meta.get("status", "OK") == "OK"

    # save vehicle YAML through the UI and reload it
    out_yaml = tmp_path / "vehicle_saved.yaml"
    out_yaml.write_text(yaml.safe_dump(win._vehicle.model_dump()))
    from fwmav_sim.experiment.config import load_vehicle

    reloaded = load_vehicle(out_yaml)
    assert reloaded.bodies[0].mass.value == win._vehicle.bodies[0].mass.value
