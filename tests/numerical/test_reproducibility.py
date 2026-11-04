"""§11.3 save/reload reproducibility (M3) and repeat-run determinism (M1).

`test_repeat_run_determinism` is real (M1 stage file §1.4); save/reload rerun
lands with the experiment record in M3.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.environment import still_air
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.simulation.fixed_body import run_fixed_body
from fwmav_sim.vehicle.schema import AeroModelSpec, VehicleDefinition, WingSpec

_PROV_EST = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}
_PROV_LIT = {"source": "literature", "reference": "Dickinson et al. 1999, Science 284:1954"}
_COEFFS = {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV_EST}


def _vehicle() -> VehicleDefinition:
    wing = WingSpec(
        name="wing_right", side="right",
        span=_param(0.3), root_chord=_param(0.08), tip_chord=_param(0.04),
        planform="trapezoidal",
        hinge_location={"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV_EST},
        pitch_axis_fraction=_param(0.0, "1"),
    )
    return VehicleDefinition(
        name="det_test",
        bodies=[{
            "name": "thorax", "mass": _param(0.01, "kg"),
            "com_offset": {"value": [0, 0, 0], "unit": "m", "provenance": _PROV_EST},
            "inertia": {"value": [[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
                        "unit": "kg m^2", "provenance": _PROV_EST},
        }],
        wings=[wing],
        aero=AeroModelSpec(
            name="quasi_steady_blade_element",
            coefficients={k: {"value": v, "unit": "1", "provenance": _PROV_LIT}
                          for k, v in _COEFFS.items()},
        ),
    )


def _kin() -> PrescribedWingKinematics:
    return PrescribedWingKinematics(
        flap_amplitude=_param(1.0, "rad"), flap_frequency=_param(12.0, "Hz"),
        pitch_amplitude=_param(0.5, "rad"), pitch_phase=_param(0.8, "rad"),
        pitch_offset=_param(0.1, "rad"), stroke_plane_angle=_param(0.15, "rad"),
    )


def test_repeat_run_determinism():
    """Same config twice => bitwise-identical time-resolved output."""
    vehicle, kin = _vehicle(), _kin()
    env = still_air()
    model = QuasiSteadyBladeElement(vehicle.aero)
    a = run_fixed_body(vehicle, kin, env, model, n_wingbeats=2, steps_per_wingbeat=50)
    b = run_fixed_body(vehicle, kin, env, model, n_wingbeats=2, steps_per_wingbeat=50)
    assert a.channel_names() == b.channel_names()
    for name in a.channel_names():
        assert np.array_equal(a[name], b[name]), name


def test_save_reload_rerun_reproduces(tmp_path):
    """fwmav-sim rerun <record>: outputs bit-identical (fixed deterministic
    pipeline; documented rule — the same holds for adaptive DOP853 here
    because tolerances and inputs are identical)."""
    import subprocess
    import sys
    from pathlib import Path

    env_out = tmp_path / "out"
    exp = Path("experiments/m2_foil_rl_sandbox.yaml").resolve()
    r1 = subprocess.run(
        [sys.executable, "-m", "fwmav_sim.cli", "run", str(exp), "--out", str(env_out)],
        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr[-2000:]
    record1 = Path(r1.stdout.strip().split("run record: ")[-1])

    r2 = subprocess.run(
        [sys.executable, "-m", "fwmav_sim.cli", "rerun", str(record1), "--out", str(env_out)],
        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr[-2000:]
    assert "bit-identical" in r2.stdout
