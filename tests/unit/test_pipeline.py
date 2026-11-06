"""Pipeline executor tests (M8 Task 0)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.control.pipeline import (
    BlockInstance,
    Pipeline,
    PipelineError,
    _toposort,
)


def _block(id_, type_, params=None, inputs=None):
    return BlockInstance.model_validate({
        "id": id_, "type": type_, "params": params or {}, "inputs": inputs or []})


def test_toposort_orders_dependencies_first():
    blocks = [
        _block("ctl", "controller_lqr", {"K": [[1, 0]], "x_trim": [0, 0]},
               [{"port": "state", "from": "est.state"}]),
        _block("est", "estimator_complementary", {},
               [{"port": "imu", "from": "imu.measurement"},
                {"port": "mocap", "from": "mocap.measurement"}]),
        _block("imu", "sensor_imu"),
        _block("mocap", "sensor_mocap"),
    ]
    order = [b.id for b in _toposort(blocks)]
    assert order.index("imu") < order.index("est")
    assert order.index("mocap") < order.index("est")
    assert order.index("est") < order.index("ctl")


def test_cycle_rejected():
    blocks = [
        _block("a", "controller_lqr", {"K": [[0] * 4]},
               [{"port": "state", "from": "b.state"}]),
        _block("b", "estimator_complementary", {},
               [{"port": "imu", "from": "c.measurement"},
                {"port": "mocap", "from": "c.measurement"}]),
        _block("c", "sensor_imu", {},
               [{"port": "state", "from": "a.channels"}]),
    ]
    with pytest.raises(PipelineError, match="cycle|mismatch|unknown"):
        Pipeline(blocks)


def test_type_mismatch_rejected():
    blocks = [
        _block("imu", "sensor_imu"),
        _block("ctl", "controller_lqr", {"K": [[0] * 6]},
               [{"port": "state", "from": "imu.measurement"}]),
    ]
    with pytest.raises(PipelineError, match="type mismatch"):
        Pipeline(blocks)


def test_unwired_required_input_rejected():
    with pytest.raises(PipelineError, match="not wired"):
        Pipeline([_block("est", "estimator_complementary",
                         {}, [{"port": "imu", "from": "imu.measurement"}]),
                    _block("imu", "sensor_imu")])


def test_unknown_block_type_rejected():
    with pytest.raises(PipelineError, match="unknown type"):
        Pipeline([_block("x", "flux_capacitor")])


def test_unknown_block_reference_rejected():
    blocks = [_block("ctl", "controller_lqr", {"K": [[0, 0]]},
                     [{"port": "state", "from": "ghost.state"}])]
    with pytest.raises(PipelineError, match="unknown block"):
        Pipeline(blocks)


def test_lqr_pipeline_matches_direct_computation():
    """A pipeline of one LQR block must reproduce the direct gain multiply —
    the executor adds no semantics of its own."""
    rng = np.random.default_rng(7)
    K = rng.normal(size=(2, 8))
    x_trim = rng.normal(size=8)
    pipe = Pipeline([_block("ctl", "controller_lqr",
                            {"K": K.tolist(), "x_trim": x_trim.tolist()})])
    x = rng.normal(size=8)
    out = pipe(x, 0.0)
    expected = -K @ (x - x_trim)
    np.testing.assert_allclose(out["commands"]["ctl"]["channels"][:2], expected)


def test_sensor_estimator_controller_chain_runs():
    """The full S3 chain: IMU + mocap -> complementary -> LQR -> mixer."""
    blocks = [
        _block("imu", "sensor_imu", {"gyro_noise_std": 1e-4, "seed": 3}),
        _block("mocap", "sensor_mocap", {"pos_noise_std": 1e-4, "seed": 2}),
        _block("est", "estimator_complementary",
               {"attitude_trust": 0.98, "state_dim": 13},
               [{"port": "imu", "from": "imu.measurement"},
                {"port": "mocap", "from": "mocap.measurement"}]),
        _block("ctl", "controller_lqr",
               {"K": (0.01 * np.ones((4, 13))).tolist()},
               [{"port": "state", "from": "est.state"}]),
        _block("mix", "mixer_wingbeat_channels", {"limit": 0.5},
               [{"port": "channels", "from": "ctl.channels"}]),
    ]
    pipe = Pipeline(blocks)
    state = np.zeros(13)
    state[3] = 1.0  # unit quat
    out0 = pipe(state, 0.0)
    out1 = pipe(state, 0.01)
    for out in (out0, out1):
        ch = out["commands"]["mix"]["channels"]
        assert ch.shape == (4,)
        assert np.all(np.isfinite(ch))
        assert np.abs(ch).max() <= 0.5  # mixer saturation respected


def test_scope_records():
    from fwmav_sim.control import blocks as blk

    step = blk.make_scope({"name": "s"})
    step({"signal": np.array(1.5)}, np.zeros(1), 0.0)
    step({"signal": np.array(2.5)}, np.zeros(1), 0.1)
    assert [v for _, v in step.log] == [1.5, 2.5]
