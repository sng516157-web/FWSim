"""M3 failure-injection test: a NaN-forcing config must stop the run visibly
with a FAILED record (never plausible-looking partial output; brief §10 M3).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fwmav_sim.simulation.run import run_experiment


def test_nan_forces_fail_visibly(tmp_path):
    exp = yaml.safe_load(open("experiments/m3_free_body_hover.yaml", encoding="utf-8"))
    exp["name"] = "m3_failure_injection"
    exp["kinematics"]["flap_amplitude"]["value"] = float("nan")  # NaN-forcing config
    exp["solver"]["max_time"] = 0.05
    cfg = tmp_path / "fail.yaml"
    exp["vehicle"] = str(Path("reference_vehicles/bio_abdomen_fwmav.yaml").resolve())
    cfg.write_text(yaml.safe_dump(exp))

    record_dir = run_experiment(cfg, tmp_path / "out")
    meta = json.loads((record_dir / "metadata.json").read_text())
    assert meta["status"] == "FAILED"
    assert "failure_message" in meta
    assert "failing_state" in meta
    # no plausible-looking partial output: the timeseries is empty
    import numpy as np

    npz = np.load(record_dir / "timeseries.npz")
    assert len(npz["t"]) == 0
