"""§11.4 mechanism-ablation comparisons (brief §6.6) — real in M3.

The three isolation cases must produce pairwise-different trajectories (the
switches actually change the plant), and the abdomen-aero flag must raise
until the Level-3 model exists (M3 stage file §3.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.simulation.free_body import IsolationSpec


def _run(case: str, tmp_path):
    from pathlib import Path

    from fwmav_sim.experiment.config import load_experiment
    from fwmav_sim.experiment.record import load_record
    from fwmav_sim.simulation.run import run_experiment

    src = Path(f"experiments/m3_isolation/{case}.yaml")
    exp = load_experiment(src)
    data = exp.model_dump()
    data["name"] = f"iso_test_{case}"
    data["solver"]["max_time"] = 0.05  # short horizon for the test
    data["vehicle"] = str(Path("reference_vehicles/bio_abdomen_fwmav.yaml").resolve())
    cfg = tmp_path / f"{case}.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump(data))
    record = run_experiment(cfg, tmp_path / "out")
    ts, meta = load_record(record)
    return ts, meta


def test_mechanism_isolation_cases(tmp_path):
    ts_base, _ = _run("baseline", tmp_path)
    ts_geom, _ = _run("geometry_only", tmp_path)
    ts_full, _ = _run("full_coupling", tmp_path)

    # pairwise different plants
    assert np.abs(ts_base["p"] - ts_geom["p"]).max() > 1e-9
    assert np.abs(ts_base["p"] - ts_full["p"]).max() > 1e-9
    assert np.abs(ts_geom["p"] - ts_full["p"]).max() > 1e-9


def test_abdomen_aero_flag_raises():
    with pytest.raises(NotImplementedError, match="Level-3"):
        IsolationSpec(abdomen_aero_enabled=True)
