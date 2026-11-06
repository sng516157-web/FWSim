"""Control channels + Gymnasium env tests (M7 Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.control.channels import ChannelCommand, ChannelMixer
from fwmav_sim.experiment.config import load_experiment, load_vehicle


def _kin_map():
    exp = load_experiment("experiments/m1_fixed_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    return {w.name: exp.kinematics for w in vehicle.wings}


def test_mixer_thrust_scales_all_wings():
    kin = _kin_map()
    base_amp = kin["wing_right"].flap_amplitude.value
    out = ChannelMixer(list(kin)).apply(kin, ChannelCommand(thrust=0.1))
    assert out["wing_right"].flap_amplitude.value == pytest.approx(base_amp * 1.1)
    assert out["wing_left"].flap_amplitude.value == pytest.approx(base_amp * 1.1)
    # inputs untouched (pure function)
    assert kin["wing_right"].flap_amplitude.value == pytest.approx(base_amp)


def test_mixer_roll_is_differential_and_pitch_shifts():
    kin = _kin_map()
    base_amp = kin["wing_right"].flap_amplitude.value
    base_off = kin["wing_right"].pitch_offset.value
    out = ChannelMixer(list(kin)).apply(kin, ChannelCommand(roll=0.05, pitch=0.02))
    assert out["wing_right"].flap_amplitude.value == pytest.approx(base_amp * 1.05)
    assert out["wing_left"].flap_amplitude.value == pytest.approx(base_amp * 0.95)
    assert out["wing_right"].pitch_offset.value == pytest.approx(base_off + 0.02)


gym = pytest.importorskip("gymnasium", reason="gymnasium extra not installed")
from gymnasium.utils import env_checker  # noqa: E402  (submodule, not re-exported)


def test_gym_env_rollout_and_interface():
    from fwmav_sim.control.gym_env import make_hover_env

    env = make_hover_env("experiments/m3_free_body_hover.yaml", max_time_s=0.15)
    env_checker.check_env(env, skip_render_check=True)
    obs, _ = env.reset()
    assert obs.shape == (13,)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample() * 0.0)  # zero command = trim-ish
        assert np.all(np.isfinite(obs))
    assert isinstance(reward, float)
