"""Canned pipeline block library (M8 Task 1).

Each ``make_<type>(params)`` factory returns a step callable
``step(inputs, state, t) -> {output_port: value}``. Blocks are stateful
where physics demands it (IMU finite-difference accel, complementary filter,
PID integrator, actuator lag); state lives in closures, never in module
globals. Sensor noise uses a per-instance seeded RNG so runs are
reproducible (provenance: noise params live in the pipeline YAML).

State layout convention (free-body plant, simulation/free_body.py):
    y = [p(3), quat(4), q_joints(nj), v_W(3), omega_B(3), qd_joints(nj)]
Blocks that read the state take explicit index params instead of hardcoding
offsets, so the same blocks serve the 8-state reduced model.
"""

from __future__ import annotations

import numpy as np


def _idx(params, key, default):
    return params.get(key, default)


# ---------------------------------------------------------------------------
# Sensors (S3: the controller sees measurements, not truth)
# ---------------------------------------------------------------------------


def make_sensor_imu(params):
    """IMU sim: gyro = omega_B + noise; accel = d(v_W)/dt + noise (finite
    difference of the velocity state; specific force in the world frame —
    the reduced model has no attitude-frame accel channel).

    params: gyro_idx (default [10,11,13) -> slice(10,13)), vel_idx,
            gyro_noise_std, accel_noise_std, seed
    """
    gyro_idx = _idx(params, "gyro_idx", [10, 11, 12])
    vel_idx = _idx(params, "vel_idx", [7, 8, 9])
    g_std = float(params.get("gyro_noise_std", 0.0))
    a_std = float(params.get("accel_noise_std", 0.0))
    rng = np.random.default_rng(params.get("seed", 0))
    prev = {"v": None, "t": None}

    def step(inputs, state, t):
        gyro = np.asarray(state[gyro_idx], float) + rng.normal(0.0, g_std, 3)
        v = np.asarray(state[vel_idx], float)
        if prev["v"] is None or t <= prev["t"]:
            accel = np.zeros(3)
        else:
            accel = (v - prev["v"]) / (t - prev["t"])
        prev["v"], prev["t"] = v.copy(), t
        accel = accel + rng.normal(0.0, a_std, 3)
        return {"measurement": np.concatenate([gyro, accel])}

    return step


def make_sensor_mocap(params):
    """Motion capture: position + quaternion with noise and an optional
    update-rate hold (returns the last sample between updates).

    params: pos_idx (default [0..2]), quat_idx (default [3..6]),
            pos_noise_std, att_noise_std, rate_hz (None = every call), seed
    """
    pos_idx = _idx(params, "pos_idx", [0, 1, 2])
    quat_idx = _idx(params, "quat_idx", [3, 4, 5, 6])
    p_std = float(params.get("pos_noise_std", 0.0))
    q_std = float(params.get("att_noise_std", 0.0))
    rate = params.get("rate_hz")
    rng = np.random.default_rng(params.get("seed", 1))
    hold = {"next_t": 0.0, "sample": None}

    def step(inputs, state, t):
        if hold["sample"] is not None and rate and t < hold["next_t"]:
            return {"measurement": hold["sample"]}
        pos = np.asarray(state[pos_idx], float) + rng.normal(0.0, p_std, 3)
        quat = np.asarray(state[quat_idx], float) + rng.normal(0.0, q_std, 4)
        quat = quat / np.linalg.norm(quat)
        hold["sample"] = np.concatenate([pos, quat])
        if rate:
            hold["next_t"] = t + 1.0 / float(rate)
        return {"measurement": hold["sample"]}

    return step


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def make_estimator_complementary(params):
    """Complementary filter: attitude = trust*mocap_quat + (1-trust)*gyro
    integral; position = mocap position; velocity = alpha-filtered finite
    difference of position. Output is a state estimate with the plant's
    layout (p, quat from sensors; v, omega filtered).

    params: attitude_trust (0..1, mocap weight), vel_alpha (0..1),
            state_layout: {p: [0,3), quat: [3,7), v: [7+nj,10+nj),
                           omega: [10+nj,13+nj)} as explicit index lists
    """
    trust = float(params.get("attitude_trust", 0.98))
    alpha = float(params.get("vel_alpha", 0.7))
    layout = params.get("state_layout", {})
    p_idx = layout.get("p", [0, 1, 2])
    q_idx = layout.get("quat", [3, 4, 5, 6])
    v_idx = layout.get("v", [7, 8, 9])
    w_idx = layout.get("omega", [10, 11, 12])
    n = int(params.get("state_dim", 13))
    est = {"p": None, "v": np.zeros(3), "t": None, "quat": None}

    def step(inputs, state, t):
        imu = np.asarray(inputs["imu"], float)
        mocap = np.asarray(inputs["mocap"], float)
        gyro, m_pos, m_quat = imu[:3], mocap[:3], mocap[3:7]
        if est["quat"] is None:
            est["quat"] = m_quat.copy()
            est["p"] = m_pos.copy()
            est["t"] = t
        dt = max(t - est["t"], 1e-9)
        # gyro integral (first order, world-frame quat kinematics)
        w = gyro
        q = est["quat"]
        q_dot = 0.5 * np.array([
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0]])
        q_gyro = q + q_dot * dt
        q_gyro /= np.linalg.norm(q_gyro)
        est["quat"] = trust * m_quat + (1.0 - trust) * q_gyro
        est["quat"] /= np.linalg.norm(est["quat"])
        v_raw = (m_pos - est["p"]) / dt
        est["v"] = alpha * est["v"] + (1.0 - alpha) * v_raw
        est["p"] = m_pos.copy()
        est["t"] = t
        out = np.zeros(n)
        out[p_idx] = est["p"]
        out[q_idx] = est["quat"]
        out[v_idx] = est["v"]
        out[w_idx] = gyro
        return {"state": out}

    return step


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def make_controller_pid(params):
    """PID on a scalar error -> one of the 4 channels (default: thrust).

    params: state_idx, setpoint, kp, ki, kd, channel (0..3), i_limit
    """
    idx = int(params["state_idx"])
    sp = float(params.get("setpoint", 0.0))
    kp, ki, kd = (float(params.get(k, 0.0)) for k in ("kp", "ki", "kd"))
    chan = int(params.get("channel", 0))
    i_lim = float(params.get("i_limit", 1e6))
    mem = {"i": 0.0, "prev_e": None, "t": None}

    def step(inputs, state, t):
        e = sp - float(state[idx])
        if mem["t"] is None:
            dt, de = 0.0, 0.0
        else:
            dt = max(t - mem["t"], 1e-9)
            de = (e - mem["prev_e"]) / dt
        mem["i"] = np.clip(mem["i"] + e * dt, -i_lim, i_lim)
        mem["prev_e"], mem["t"] = e, t
        u = kp * e + ki * mem["i"] + kd * de
        out = np.zeros(4)
        out[chan] = u
        return {"channels": out}

    return step


def make_controller_lqr(params):
    """Full-state LQR: channels = -K (x - x_trim), with optional per-channel
    saturation. params: K (m x n nested list), x_trim (n list),
    state_indices (optional subset map), limits (optional m list).
    """
    K = np.asarray(params["K"], float)
    x_trim = np.asarray(params.get("x_trim", np.zeros(K.shape[1])), float)
    idx = params.get("state_indices")
    limits = params.get("limits")
    limits = np.asarray(limits, float) if limits is not None else None

    def step(inputs, state, t):
        del state  # controllers act on the WIRED (possibly estimated) state
        x = np.asarray(inputs["state"], float)
        if idx is not None:
            x = x[idx]
        u = -K @ (x - x_trim)
        if limits is not None:
            u = np.clip(u, -limits, limits)
        out = np.zeros(4)
        out[: len(u)] = u
        return {"channels": out}

    return step


# ---------------------------------------------------------------------------
# Mixer / actuator / sink
# ---------------------------------------------------------------------------


def make_summer_channels(params):
    """Elementwise sum of two channel vectors (e.g. altitude PID + attitude
    PID before the mixer)."""
    def step(inputs, state, t):
        return {"channels": np.asarray(inputs["channels_a"], float)
                + np.asarray(inputs["channels_b"], float)}

    return step


def make_mixer_wingbeat_channels(params):
    """Channel pass-through with saturation — the ChannelCommand contract
    (control/channels.py). Exists as a block so the DAG has an explicit
    mixer stage (flappy's wing-beat-modulation convention)."""
    lim = float(params.get("limit", 0.5))

    def step(inputs, state, t):
        return {"channels": np.clip(np.asarray(inputs["channels"], float), -lim, lim)}

    return step


def make_actuator_torque(params):
    """First-order-lag torque actuator driven by one channel:
    tau_dot = (gain * u - tau) / tau_s, saturated at tau_max.
    params: channel (0..3), gain (N m per unit channel), tau_s, tau_max
    """
    chan = int(params.get("channel", 1))
    gain = float(params.get("gain", 1e-4))
    tau_s = float(params.get("tau_s", 0.02))
    tau_max = float(params.get("tau_max", 1e-3))
    mem = {"tau": 0.0, "t": None}

    def step(inputs, state, t):
        u = float(np.asarray(inputs["channels"], float)[chan])
        if mem["t"] is not None:
            dt = max(t - mem["t"], 1e-9)
            mem["tau"] += dt * (gain * u - mem["tau"]) / tau_s
        mem["tau"] = float(np.clip(mem["tau"], -tau_max, tau_max))
        mem["t"] = t
        return {"torque": mem["tau"]}

    return step


def make_scope(params):
    """Sink that records its input per call into a list (inspection/tests).
    params: name."""
    name = params.get("name", "scope")
    log: list[tuple[float, float]] = []

    def step(inputs, state, t):
        log.append((t, float(np.asarray(inputs["signal"], float))))
        return {}

    step.log = log  # tests read scope logs here
    step.name = name
    return step
