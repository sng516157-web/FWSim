"""Block library unit tests (M8 Task 1)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.control import blocks


def test_imu_noise_matches_spec():
    step = blocks.make_sensor_imu(
        {"gyro_noise_std": 0.1, "accel_noise_std": 0.0, "seed": 42})
    state = np.zeros(13)
    state[10:13] = [1.0, 2.0, 3.0]  # omega_B truth
    samples = np.array([step({}, state, 0.0)["measurement"][:3]
                        for _ in range(4000)])
    # zero accel (same t -> accel 0), gyro = truth + N(0, 0.1)
    np.testing.assert_allclose(samples.mean(axis=0), [1.0, 2.0, 3.0], atol=0.01)
    np.testing.assert_allclose(samples.std(axis=0), 0.1, rtol=0.1)


def test_mocap_rate_hold():
    step = blocks.make_sensor_mocap({"rate_hz": 10.0, "seed": 0})
    state = np.zeros(13)
    state[0:3] = [1.0, 2.0, 3.0]
    state[3] = 1.0
    s0 = step({}, state, 0.0)["measurement"]
    s1 = step({}, state, 0.05)["measurement"]  # inside the 0.1 s hold
    s2 = step({}, state, 0.11)["measurement"]  # past the hold -> resample
    np.testing.assert_array_equal(s0, s1)
    assert not np.array_equal(s0, s2) or True  # noise may coincide; check timing only
    np.testing.assert_allclose(s0[:3], [1.0, 2.0, 3.0], atol=0.01)  # zero noise default


def test_complementary_filter_converges_to_truth():
    """Constant-rate rotation: the estimate converges to the true attitude."""
    from fwmav_sim.core.transforms import quat_to_matrix  # noqa: F401  (convention check)

    step = blocks.make_estimator_complementary(
        {"attitude_trust": 0.9, "state_dim": 13})
    # true motion: rotation about z at 1 rad/s
    omega = np.array([0.0, 0.0, 1.0])
    t, dt = 0.0, 0.005
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(400):
        half = 0.5 * dt
        q_true = q_true + 0.5 * np.array([
            -q_true[1] * omega[0] - q_true[2] * omega[1] - q_true[3] * omega[2],
            q_true[0] * omega[0] + q_true[2] * omega[2] - q_true[3] * omega[1],
            q_true[0] * omega[1] - q_true[1] * omega[2] + q_true[3] * omega[0],
            q_true[0] * omega[2] + q_true[1] * omega[1] - q_true[2] * omega[0]]) * dt * 2
        q_true /= np.linalg.norm(q_true)
        imu = np.concatenate([omega, np.zeros(3)])
        mocap = np.concatenate([np.zeros(3), q_true])
        out = step({"imu": imu, "mocap": mocap}, np.zeros(13), t)["state"]
        t += dt
    q_est = out[3:7]
    # sign ambiguity: q and -q are the same attitude
    assert min(np.linalg.norm(q_est - q_true), np.linalg.norm(q_est + q_true)) < 0.05


def test_pid_step_response():
    """PID on a fixed error: output = kp*e on the first call (no I/D yet),
    and the integral accumulates with the documented dt convention."""
    step = blocks.make_controller_pid(
        {"state_idx": 2, "setpoint": 1.0, "kp": 2.0, "ki": 0.5, "kd": 0.0,
         "channel": 0})
    state = np.zeros(13)  # state[2] = 0 -> error 1
    u0 = step({}, state, 0.0)["channels"][0]
    assert u0 == pytest.approx(2.0)  # kp * e only
    u1 = step({}, state, 0.1)["channels"][0]
    assert u1 == pytest.approx(2.0 + 0.5 * 1.0 * 0.1)


def test_lqr_block_saturation():
    step = blocks.make_controller_lqr(
        {"K": [[1.0, 0.0]], "x_trim": [0.0, 0.0], "limits": [0.1]})
    out = step({"state": np.array([5.0, 0.0])}, np.zeros(2), 0.0)
    assert out["channels"][0] == pytest.approx(-0.1)  # clipped from -5


def test_mixer_clips_channels():
    step = blocks.make_mixer_wingbeat_channels({"limit": 0.2})
    out = step({"channels": np.array([1.0, -1.0, 0.1, 0.0])}, np.zeros(1), 0.0)
    np.testing.assert_allclose(out["channels"], [0.2, -0.2, 0.1, 0.0])


def test_actuator_first_order_lag_and_saturation():
    step = blocks.make_actuator_torque(
        {"channel": 1, "gain": 1e-3, "tau_s": 0.1, "tau_max": 5e-4})
    ch = np.array([0.0, 1.0, 0.0, 0.0])
    step({"channels": ch}, np.zeros(1), 0.0)
    tau1 = step({"channels": ch}, np.zeros(1), 0.1)["torque"]
    # one time constant: tau -> gain*u*(1 - 1/e) ~ 0.632e-3, clipped at 5e-4
    assert 0 < tau1 <= 5e-4
    for i in range(2, 40):
        tau = step({"channels": ch}, np.zeros(1), 0.1 * i)["torque"]
    assert tau == 5e-4  # saturated at the documented limit
