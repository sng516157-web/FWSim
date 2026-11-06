"""GL live session (M9 T5): step the free-body plant and render into the
Qt-native GLSceneWidget — the LiveSession equivalent without MeshCat.

Same contract as viz/live.LiveSession: step_frame(advance_s) advances and
renders one frame; wall pacing is the caller's job; recording via
SessionRecorder produces the standard run-record schema on stop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class GLLiveSession:
    def __init__(self, experiment_path: str | Path, gl_widget,
                 force_scale: float = 5.0, record: bool = True,
                 out_root: str | Path = "outputs"):
        from scipy.integrate import RK45

        from fwmav_sim.experiment.config import kinematics_map, load_experiment, load_vehicle
        from fwmav_sim.simulation.run import build_free_body_plant

        experiment_path = Path(experiment_path)
        exp = load_experiment(experiment_path)
        if exp.kind != "free_body":
            raise ValueError(
                f"live requires a free_body experiment (got {exp.kind!r})")
        self.plant, y0, vehicle_path = build_free_body_plant(exp, experiment_path)
        self.exp = exp
        self.t = 0.0
        self.y = y0
        period = 1.0 / self.plant.kinematics.flap_frequency.value
        self._stepper = RK45(self.plant.rhs, 0.0, y0, t_bound=np.inf,
                             rtol=exp.solver.rtol, atol=exp.solver.atol,
                             max_step=period / exp.solver.steps_per_wingbeat)
        self._gl = gl_widget
        self._force_scale = force_scale
        self._out_root = out_root
        self._kin_map = kinematics_map(exp, self.plant.vehicle)

        from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
        from fwmav_sim.environment.spec import environment_from_spec
        from fwmav_sim.viz.meshcat_scene import _FreeBodyReplayCtx

        self._env = environment_from_spec(exp.environment)
        self._model = QuasiSteadyBladeElement(self.plant.vehicle.aero)
        self._ctx_cls = _FreeBodyReplayCtx

        self._recorder = None
        if record:
            from fwmav_sim.experiment.record import SessionRecorder

            self._recorder = SessionRecorder(
                name=f"live_{exp.name}", experiment_yaml=experiment_path,
                vehicle_yaml=vehicle_path,
                ts_meta={"kind": "free_body", "status": "OK", "session": "live_gl",
                         "solver": {"integrator": "RK45", "rtol": exp.solver.rtol,
                                    "atol": exp.solver.atol}},
                extra_meta={"validity": {"assumption_ids": ["A-007"],
                                         "warnings": ["live GL session record; same "
                                                      "schema as a batch run record"]}})

    def _frame_ctx(self):
        """A one-frame replay context over the CURRENT state (frame_geometry
        consumes the same channels as a record)."""
        from fwmav_sim.core.timeseries import TimeSeries

        nj = self.plant.nj
        y = self.y
        channels = {
            "p": y[:3][None, :],
            "quat": y[3:7][None, :],
            "q_joints": y[7:7 + nj][None, :],
            "v_W": y[7 + nj:10 + nj][None, :],
            "omega_B": y[10 + nj:13 + nj][None, :],
        }
        ts = TimeSeries(t=np.array([self.t]), channels=channels, meta={})
        return self._ctx_cls(ts=ts, vehicle=self.plant.vehicle, env=self._env,
                             model=self._model, kin_map=self._kin_map,
                             art=None, state=None, force_scale=self._force_scale)

    def step_frame(self, advance_s: float) -> None:
        t_end = self.t + advance_s
        while self._stepper.t < t_end:
            self._stepper.step()
        self.t = self._stepper.t
        self.y = self._stepper.y
        if self._recorder is not None:
            self._recorder.add_sample(self.t, self._frame_channels())
        from fwmav_sim.viz.gl_frames import frame_geometry

        meshes, arrows = frame_geometry(self._frame_ctx(), 0)
        self._gl.set_wing_meshes(meshes)
        self._gl.set_force_arrows(arrows)
        self._gl.update()

    def _frame_channels(self) -> dict[str, np.ndarray]:
        nj = self.plant.nj
        y = self.y
        return {"p": y[:3], "quat": y[3:7], "q_joints": y[7:7 + nj],
                "v_W": y[7 + nj:10 + nj], "omega_B": y[10 + nj:13 + nj],
                "qd_joints": y[13 + nj:13 + 2 * nj]}

    def save_record(self):
        if self._recorder is None:
            return None
        return self._recorder.save(self._out_root)
