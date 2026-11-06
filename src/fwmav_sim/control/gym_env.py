"""Gymnasium environment wrapper for the free-body hover task (M7 Task 4).

FlapSimEnv: the free_body plant as a Gymnasium env. Observation: body state
(position, attitude euler, rates) + wingbeat phase. Action: wing-beat channel
deltas [thrust, roll, pitch, yaw] applied at each wingbeat boundary via
ChannelMixer (control/channels.py). Reward: hover keeping (documented, not
tuned: -(position error + attitude error)).

gymnasium is an optional dependency; importing this module without it raises
a clear error and tests skip via importorskip.
"""

from __future__ import annotations

import numpy as np


def _require_gym():
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:
        raise ImportError("the RL environment needs the optional dependency: "
                          "pip install fwmav-sim[rl]") from exc
    return gym, spaces


def make_hover_env(experiment_path, max_time_s: float = 2.0,
                   step_per_beat: int = 1):
    """Build the hover env for a free_body experiment YAML."""
    gym, spaces = _require_gym()
    from pathlib import Path

    from fwmav_sim.control.channels import ChannelCommand, ChannelMixer
    from fwmav_sim.experiment.config import load_experiment, load_vehicle
    from fwmav_sim.simulation.run import build_free_body_plant

    experiment_path = Path(experiment_path)

    class FlapSimEnv(gym.Env):
        """Hover task: stay at the origin, level attitude."""

        metadata = {"render_modes": []}

        def __init__(self):
            exp = load_experiment(experiment_path)
            self._exp_path = experiment_path
            self._max_time = max_time_s
            vehicle = load_vehicle(experiment_path.parent / exp.vehicle)
            self._mixer = ChannelMixer([w.name for w in vehicle.wings])
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            # obs: p(3), euler(3), v(3), omega(3), phase(1)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,),
                                                dtype=np.float64)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            exp = load_experiment(self._exp_path)
            self.plant, self._y, _ = build_free_body_plant(exp, self._exp_path)
            self._kin_now = {w.name: self.plant.kinematics_map[w.name]
                             for w in self.plant.vehicle.wings}
            self._t = 0.0
            return self._obs(), {}

        def _obs(self):
            from fwmav_sim.core.transforms import quat_to_matrix

            y = self._y
            R = quat_to_matrix(y[3:7])
            euler = np.array([np.arctan2(R[2, 1], R[2, 2]),
                              -np.arcsin(np.clip(R[2, 0], -1, 1)),
                              np.arctan2(R[1, 0], R[0, 0])])
            f = self.plant.kinematics.flap_frequency.value
            phase = (self._t * f) % 1.0
            return np.concatenate([y[0:3], euler, y[7 + self.plant.nj:10 + self.plant.nj],
                                   y[10 + self.plant.nj:13 + self.plant.nj],
                                   [phase]])

        def step(self, action):
            cmd = ChannelCommand(*np.asarray(action, dtype=float))
            self._kin_now = self._mixer.apply(self._kin_now, cmd)
            for name, kin in self._kin_now.items():
                self.plant.kinematics_map[name] = kin
            # integrate one wingbeat
            from scipy.integrate import solve_ivp

            f = self.plant.kinematics.flap_frequency.value
            sol = solve_ivp(self.plant.rhs, (self._t, self._t + 1.0 / f), self._y,
                            method="RK45", rtol=1e-7, atol=1e-9)
            self._y = sol.y[:, -1]
            self._t = sol.t[-1]
            obs = self._obs()
            reward = -float(np.linalg.norm(obs[0:3]) + np.linalg.norm(obs[3:6]))
            terminated = bool(np.abs(obs[0:3]).max() > 1.0 or self._t >= self._max_time)
            return obs, reward, terminated, False, {}

    return FlapSimEnv()
