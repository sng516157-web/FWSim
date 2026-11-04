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


def test_abdomen_aero_flag_requires_vehicle_spec():
    """Level 3 (post-M5 item 4): the flag needs an abdomen_aero block and an
    articulated abdomen; without them the plant raises a clear error."""
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.simulation.free_body import FreeBodyPlant
    from fwmav_sim.vehicle.schema import VehicleDefinition

    prov = {"source": "estimated"}
    vehicle = VehicleDefinition.model_validate({
        "name": "no_abdomen_aero",
        "bodies": [{"name": "body",
                    "mass": {"value": 0.02, "unit": "kg", "provenance": prov},
                    "com_offset": {"value": [0, 0, 0], "unit": "m", "provenance": prov},
                    "inertia": {"value": (np.eye(3) * 5e-5).tolist(),
                                "unit": "kg m^2", "provenance": prov}}],
        "joints": [], "wings": [], "aero": {"name": "none", "coefficients": {}},
    })
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.zeros(3))
    from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
    kin = PrescribedWingKinematics.model_validate({
        "flap_amplitude": {"value": 0.5, "unit": "rad", "provenance": prov},
        "flap_frequency": {"value": 10.0, "unit": "Hz", "provenance": prov},
        "pitch_amplitude": {"value": 0.3, "unit": "rad", "provenance": prov},
        "pitch_phase": {"value": 0.0, "unit": "rad", "provenance": prov},
        "pitch_offset": {"value": 0.0, "unit": "rad", "provenance": prov},
        "stroke_plane_angle": {"value": 0.0, "unit": "rad", "provenance": prov},
    })
    with pytest.raises(ValueError, match="abdomen_aero"):
        FreeBodyPlant(vehicle, kin, env, None,
                      isolation=IsolationSpec(abdomen_aero_enabled=True))
