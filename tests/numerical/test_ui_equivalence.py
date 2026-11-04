"""§11.3 M5: headless/UI equivalence (real, acceptance-level).

The same experiment YAML executed through the UI's RunWorker and through
`fwmav-sim run` must produce equal outputs (the UI adds zero physics).
Deterministic pipeline => bit-identical (documented M3 rule).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
pytest.importorskip("PySide6", reason="gui extra not installed")

from PySide6.QtWidgets import QApplication

from fwmav_sim.experiment.record import load_record


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_headless_ui_equivalence(qapp, tmp_path):
    from fwmav_sim.ui.run_panel import RunWorker

    exp = Path("experiments/m2_foil_rl_sandbox.yaml").resolve()
    out_ui = tmp_path / "ui"
    out_cli = tmp_path / "cli"

    # UI path: the run worker (same function the panel uses)
    results = {}
    worker = RunWorker(exp, out_ui)
    worker.finished_ok.connect(lambda d: results.setdefault("dir", d))
    worker.failed.connect(lambda m: results.setdefault("error", m))
    worker.run()  # synchronous in-test (no event loop needed)
    assert "dir" in results, results.get("error")

    # CLI path
    r = subprocess.run(
        [sys.executable, "-m", "fwmav_sim.cli", "run", str(exp), "--out", str(out_cli)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cli_dir = Path(r.stdout.strip().split("run record: ")[-1])

    ts_ui, _ = load_record(results["dir"])
    ts_cli, _ = load_record(cli_dir)
    assert ts_ui.channel_names() == ts_cli.channel_names()
    for name in ts_ui.channel_names():
        assert np.array_equal(ts_ui[name], ts_cli[name]), name
