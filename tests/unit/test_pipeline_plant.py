"""M8 Task 4: pipeline-driven free-body plant tests."""

from __future__ import annotations

import numpy as np

from fwmav_sim.control.channels import ChannelMixer
from fwmav_sim.control.pipeline import BlockInstance, Pipeline
from fwmav_sim.simulation.run import build_free_body_plant
from fwmav_sim.experiment.config import load_experiment
from pathlib import Path


def _lqr_pipeline(K, x_trim):
    return Pipeline([BlockInstance.model_validate({
        "id": "ctl", "type": "controller_lqr",
        "params": {"K": K, "x_trim": x_trim}, "inputs": []})])


def test_attach_pipeline_and_wingbeat_update():
    exp = load_experiment("experiments/m3_free_body_hover.yaml")
    plant, y0, _ = build_free_body_plant(exp, Path("experiments/m3_free_body_hover.yaml"))
    assert plant.pipeline is None
    f = plant.kinematics.flap_frequency.value

    # constant-gain "controller": thrust = -0.1 * z (NED: descend -> more thrust)
    pipe = _lqr_pipeline([[0.0] * 2 + [-0.1] + [0.0] * (len(y0) - 3)],
                         [0.0] * len(y0))
    base_amp = {n: k.flap_amplitude.value for n, k in plant.kinematics_map.items()}
    plant.attach_pipeline(pipe, ChannelMixer(list(plant.kinematics_map)))
    assert plant.pipeline is not None

    # before the first beat boundary crossing at t=0 the hook fires (beat 0)
    plant._update_control(0.0, y0)
    # z=0 -> channel 0 -> kinematics unchanged from baseline
    for n, k in plant.kinematics_map.items():
        assert k.flap_amplitude.value == base_amp[n]

    # descend to z=1 at t just into beat 1: thrust = +0.1 -> amplitude x1.1
    y1 = y0.copy()
    y1[2] = 1.0
    plant._update_control(1.0 / f + 1e-6, y1)
    for n, k in plant.kinematics_map.items():
        assert k.flap_amplitude.value == np.finfo(float).eps or \
            abs(k.flap_amplitude.value - 1.1 * base_amp[n]) < 1e-12

    # same beat: no re-evaluation (wingbeat-scale control, A-007)
    plant._baseline_kinematics[n].flap_amplitude.value = -999.0  # sabotage
    plant._update_control(1.0 / f + 2e-6, y1)  # same beat -> no-op
    assert plant.kinematics_map[n].flap_amplitude.value > -1.0


def test_pipeline_yaml_roundtrip():
    pipe = Pipeline.from_yaml("experiments/m8_hover_pipeline.yaml")
    state = np.zeros(17)
    state[3] = 1.0
    out = pipe(state, 0.0)
    out2 = pipe(state, 0.01)
    ch = out2["commands"]["mix"]["channels"]
    assert ch.shape == (4,)
    assert np.all(np.isfinite(ch))
