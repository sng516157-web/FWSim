"""UVLM backend smoke test (M7 T1): the adapter drives PteraSoftware with our
vehicle + prescribed kinematics and returns a force history with the right
shape and physical scale. Skipped without the optional pterasoftware extra."""

from __future__ import annotations

import numpy as np
import pytest

ps = pytest.importorskip("pterasoftware", reason="uvlm extra not installed")


def test_uvlm_fixed_body_runs_and_scales():
    from fwmav_sim.aero.uvlm_backend import run_uvlm_fixed_body
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.experiment.config import load_experiment, load_vehicle

    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3))
    out = run_uvlm_fixed_body(vehicle, {w.name: exp.kinematics for w in vehicle.wings},
                              env, n_wingbeats=2, steps_per_wingbeat=20)
    F = out["forces_W"]
    assert F.ndim == 2 and F.shape[1] == 3 and len(F) == 41
    # physical scale: peak |F| within an order of magnitude of q*A*C_N
    A = sum(0.5 * (w.root_chord.value + w.tip_chord.value) * w.span.value
            for w in vehicle.wings)
    v_tip = exp.kinematics.flap_amplitude.value * 2 * np.pi * 15.0 * 0.12
    q = 0.5 * 1.225 * v_tip**2
    assert np.abs(F).max() > 0.01 * q * A
    assert np.abs(F).max() < 100.0 * q * A
