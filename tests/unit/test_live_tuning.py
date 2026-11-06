"""Live tuning write-through tests (M9 T4)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.control.pipeline import BlockInstance, Pipeline, PipelineError


def _pid_pipeline():
    return Pipeline([BlockInstance.model_validate({
        "id": "pid", "type": "controller_pid",
        "params": {"state_idx": 2, "setpoint": 1.0, "kp": 2.0, "ki": 0.0,
                   "kd": 0.0, "channel": 0},
        "inputs": []})])


def test_set_param_retunes_running_block():
    pipe = _pid_pipeline()
    state = np.zeros(13)
    u0 = pipe(state, 0.0)["commands"]["pid"]["channels"][0]
    assert u0 == pytest.approx(2.0)  # kp * (sp - 0)
    pipe.set_param("pid", "kp", 5.0)
    u1 = pipe(state, 0.01)["commands"]["pid"]["channels"][0]
    assert u1 == pytest.approx(5.0)


def test_set_param_loud_on_unknown():
    pipe = _pid_pipeline()
    with pytest.raises(PipelineError, match="no block"):
        pipe.set_param("ghost", "kp", 1.0)
    with pytest.raises(KeyError, match="no live param"):
        pipe.set_param("pid", "ghost", 1.0)


def test_mixer_limit_live():
    pipe = Pipeline([
        BlockInstance.model_validate({"id": "pid", "type": "controller_pid",
                                      "params": {"state_idx": 0, "setpoint": 1.0,
                                                 "kp": 10.0},
                                      "inputs": []}),
        BlockInstance.model_validate({"id": "mix", "type": "mixer_wingbeat_channels",
                                      "params": {"limit": 0.2},
                                      "inputs": [{"port": "channels",
                                                  "from": "pid.channels"}]}),
    ])
    state = np.zeros(13)
    ch = pipe(state, 0.0)["commands"]["mix"]["channels"]
    assert ch[0] == pytest.approx(0.2)  # clipped
    pipe.set_param("mix", "limit", 0.9)  # retune live
    ch = pipe(state, 0.01)["commands"]["mix"]["channels"]
    assert ch[0] == pytest.approx(0.9)


def test_live_params_listing():
    pipe = _pid_pipeline()
    lp = pipe.live_params()
    assert lp["pid"]["kp"] == 2.0
    pipe.set_param("pid", "kp", 3.5)
    assert pipe.live_params()["pid"]["kp"] == 3.5
