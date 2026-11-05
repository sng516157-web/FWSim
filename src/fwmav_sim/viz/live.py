"""Live free-body simulation (streaming, not replay).

The CLI ``view`` command replays records or loops prescribed kinematics;
``live`` integrates the free-body plant continuously and streams frames to
MeshCat at a wall-clock-paced rate. Same physics as ``run`` (the plant comes
from simulation/run.build_free_body_plant — one physics path), no new models.

Usage: fwmav-sim live experiments/m3_free_body_hover.yaml --speed 0.1
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class LiveSession:
    """Integrate-and-stream session over a free-body experiment.

    step_frame(advance_sim_s): advance the plant by advance_sim_s of simulated
    time and render one frame. Wall-clock pacing is the caller's job (CLI loop
    or GUI timer) — the session itself never sleeps, so it is testable
    headlessly.

    record=True (default): the session IS the simulation and every solver
    step is recorded via experiment/record.SessionRecorder — stopping the
    session yields a full run record identical in schema to `fwmav-sim run`.
    """

    def __init__(self, experiment_path: str | Path, vis, force_scale: float = 5.0,
                 record: bool = True, out_root: str | Path = "outputs"):
        from fwmav_sim.core.timeseries import TimeSeries
        from fwmav_sim.experiment.config import load_experiment
        from fwmav_sim.simulation.run import build_free_body_plant
        from fwmav_sim.viz.meshcat_scene import build_free_body_scene

        experiment_path = Path(experiment_path)
        exp = load_experiment(experiment_path)
        if exp.kind != "free_body":
            raise ValueError(
                f"live requires a free_body experiment (got {exp.kind!r}); "
                "fixed_body kinematics animate with `fwmav-sim view <experiment>`")
        self.plant, y0, vehicle_path = build_free_body_plant(exp, experiment_path)
        self._vehicle = self.plant.vehicle
        self.t = 0.0
        self.y = y0
        from scipy.integrate import RK45

        period = 1.0 / self.plant.kinematics.flap_frequency.value
        self._stepper = RK45(self.plant.rhs, 0.0, y0, t_bound=np.inf,
                             rtol=exp.solver.rtol, atol=exp.solver.atol,
                             max_step=period / exp.solver.steps_per_wingbeat)
        self._vis = vis
        self._TimeSeries = TimeSeries
        self._ctx = build_free_body_scene(vis, self._frame_ts(0.0, y0),
                                          self._vehicle, exp, force_scale)
        # session recording (identical schema to run records)
        self._recorder = None
        self._out_root = out_root
        if record:
            from fwmav_sim.experiment.record import SessionRecorder

            self._recorder = SessionRecorder(
                name=f"live_{exp.name}", experiment_yaml=experiment_path,
                vehicle_yaml=vehicle_path,
                ts_meta={"kind": "free_body", "status": "OK", "session": "live",
                         "solver": {"integrator": "RK45", "rtol": exp.solver.rtol,
                                    "atol": exp.solver.atol},
                         "abdomen_mode": self.plant.control.mode,
                         "base_mode": self.plant.base_mode,
                         "release_time": self.plant.release_time,
                         "assumptions": ["A-001", "A-002", "A-003", "A-004", "A-005",
                                         "A-006", "A-008", "A-009"]},
                extra_meta={"validity": {"assumption_ids": ["A-007"],
                                         "warnings": ["live session record; same "
                                                      "schema as a batch run record"]}})

    def step_frame(self, advance_s: float) -> None:
        """Advance the sim by advance_s (simulated seconds) and render."""
        from fwmav_sim.viz.meshcat_scene import render_free_body_frame

        t_end = self.t + advance_s
        while self._stepper.t < t_end:
            self._stepper.step()
            if self._stepper.status != "running":
                raise RuntimeError("live sim solver stopped: "
                                   + (self._stepper.status or ""))
            if self._recorder is not None:
                self._recorder.add_sample(self._stepper.t,
                                          self._sample_channels(self._stepper.t,
                                                                self._stepper.y))
        self.t = self._stepper.t
        self.y = self._stepper.y
        self._ctx.ts = self._frame_ts(self.t, self.y)
        render_free_body_frame(self._vis, self._ctx, 0)

    def _sample_channels(self, t: float, y: np.ndarray) -> dict[str, np.ndarray]:
        """One recorded sample: state + plant diagnostics (same channels as a
        batch free_body record)."""
        nj = self.plant.nj
        self.plant.rhs(t, y)  # recompute diagnostics for this state
        p, quat = y[0:3], y[3:7]
        qm = np.concatenate([p, quat, y[7:7 + nj]])
        com = self.plant.model.com(qm)
        ch: dict[str, np.ndarray] = {
            "p": p.copy(), "quat": quat.copy(),
            "q_joints": y[7:7 + nj].copy(), "v_W": y[7 + nj:10 + nj].copy(),
            "omega_B": y[10 + nj:13 + nj].copy(),
            "qd_joints": y[13 + nj:13 + 2 * nj].copy(),
            "alpha_commanded": np.array([float(self.plant.last_alpha_cmd)]),
            "actuator_saturated": np.array([float(self.plant.last_saturated)]),
            "reaction_torques": np.asarray(self.plant.last_reaction, dtype=float),
            "com_W": np.asarray(com, dtype=float),
        }
        for w in self.plant.vehicle.wings:
            rec = self.plant.last_F_wings.get(w.name)
            if rec:
                for k in ("F_trans", "F_rot", "F_am"):
                    ch[f"{w.name}/{k}"] = np.asarray(rec[k], dtype=float)
                if "F_unsteady" in rec:
                    ch[f"{w.name}/F_unsteady"] = np.asarray(rec["F_unsteady"], dtype=float)
                ch[f"{w.name}/F_aero_total"] = (np.asarray(rec["F_trans"])
                                                + np.asarray(rec["F_rot"])
                                                + np.asarray(rec["F_am"])
                                                + np.asarray(rec.get("F_unsteady", 0.0)))
                ch[f"{w.name}/lever_arm"] = np.asarray(rec["r_app_W"]) - com
            fi = self.plant.last_F_wing_inertia.get(w.name)
            if fi is not None:
                ch[f"{w.name}/F_wing_inertial"] = np.asarray(fi, dtype=float)
        return ch

    def save_record(self):
        """Stop recording and write the session record (run-record schema)."""
        if self._recorder is None:
            return None
        out = self._recorder.save(self._out_root)
        return out

    def _frame_ts(self, t, y):
        nj = self.plant.nj
        return self._TimeSeries(
            t=np.array([t]),
            channels={
                "p": y[0:3][None, :], "quat": y[3:7][None, :],
                "q_joints": y[7:7 + nj][None, :], "v_W": y[7 + nj:10 + nj][None, :],
                "omega_B": y[10 + nj:13 + nj][None, :],
                "qd_joints": y[13 + nj:13 + 2 * nj][None, :],
            },
            meta={"kind": "free_body"},
        )


def serve_live_dynamics(experiment_path: str | Path, speed: float = 0.1,
                        force_scale: float = 5.0, record: bool = True) -> str:
    """Integrate + stream to MeshCat until Ctrl-C; the session is recorded and
    saved as a run record on stop (default; --no-record to disable).
    Returns the URL."""
    import meshcat

    vis = meshcat.Visualizer()
    session = LiveSession(experiment_path, vis, record=record)
    url = vis.url()
    print(f"MeshCat live dynamics: {url}   (Ctrl-C to stop; "
          f"speed {speed}x realtime)")
    frame_wall = 1.0 / 30.0  # render at ~30 Hz wall clock
    try:
        while True:
            t0 = time.perf_counter()
            session.step_frame(frame_wall * speed)
            dt = time.perf_counter() - t0
            time.sleep(max(frame_wall - dt, 0.0))
    except KeyboardInterrupt:
        pass
    rec = session.save_record()
    if rec is not None:
        print(f"session record: {rec}  ({session._recorder.n_samples} samples)")
    return url
