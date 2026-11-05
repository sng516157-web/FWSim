"""3D view (M5 Task 4; playback added post-M5): MeshCat in QWebEngineView,
driven by a meshcat server in the same process (DR-002). Content comes from
the vehicle definition and run-record channels — CoM, lever arms, force
overlays — never hard-coded.

Playback: ``set_record`` builds a cached replay context (free_body records
replay the recorded trajectory via viz/meshcat_scene.build_free_body_replay;
other records animate the prescribed wing kinematics via
build_kinematics_replay) and a QTimer steps frames while playing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAVE_WEBENGINE = True
except ImportError:
    _HAVE_WEBENGINE = False

_TICK_MS = 30  # playback timer period
_SPEEDS = ("0.05", "0.1", "0.25", "0.5", "1.0")


class Scene3DPanel(QWidget if _HAVE_QT else object):
    """MeshCat browser view + scene content management + run playback."""

    def __init__(self, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the 3D panel")
        super().__init__(parent)
        self._vis = None
        self._url: str | None = None
        self._replay = None  # ("free_body", ctx, n) | ("kinematics", state, period, model, env)
        self._frame = 0.0  # fractional frame index while playing
        self._playing = False

        layout = QVBoxLayout(self)
        if _HAVE_WEBENGINE:
            self._web = QWebEngineView()
            layout.addWidget(self._web)
        else:
            from PySide6.QtWidgets import QLabel as _L

            layout.addWidget(_L("QtWebEngine unavailable; open the MeshCat "
                                "URL in a browser (printed on run)."))
            self._web = None

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.setToolTip(
            "Animate the loaded run record.\n"
            "free_body runs: the recorded flight trajectory. fixed_body runs:\n"
            "the prescribed wing kinematics (the body is clamped on the rig).")
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setToolTip("Scrub through the record frame by frame")
        self.frame_slider.sliderReleased.connect(self._slider_released)
        controls.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel("–")
        controls.addWidget(self.frame_label)
        controls.addWidget(QLabel("speed"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(_SPEEDS)
        self.speed_combo.setCurrentText("0.1")
        self.speed_combo.setToolTip("Playback speed as a fraction of real time "
                                    "(0.05 = 20x slow motion)")
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)

    @property
    def url(self) -> str | None:
        return self._url

    def start_server(self) -> str:
        """Start the shared MeshCat server and show it."""
        import meshcat

        self._vis = meshcat.Visualizer()
        self._url = self._vis.url()
        if self._web is not None:
            self._web.setUrl(QUrl(self._url))
        return self._url

    # --- record / experiment loading ------------------------------------------

    def set_experiment(self, experiment_path) -> None:
        """Preview an experiment YAML in 3D: vehicle + prescribed wing
        kinematics (paused at frame 0; Play animates the wingbeat)."""
        self.stop()
        experiment_path = Path(experiment_path)
        if self._vis is None:
            self.start_server()
        from fwmav_sim.viz.meshcat_scene import build_kinematics_replay

        state, period, model, env = build_kinematics_replay(self._vis, experiment_path)
        self._replay = ("kinematics", state, period, model, env, 120)
        self._frame = 0.0
        self.frame_slider.setRange(0, 120 - 1)
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self._render_frame(0)

    def set_record(self, record_dir) -> None:
        """Load a run record for playback and show frame 0 (paused)."""
        self.stop()
        record_dir = Path(record_dir)
        if self._vis is None:
            self.start_server()
        meta = json.loads((record_dir / "metadata.json").read_text(encoding="utf-8"))
        kind = meta.get("time_series_meta", {}).get("kind")
        from fwmav_sim.viz.meshcat_scene import (
            build_free_body_replay,
            build_kinematics_replay,
        )

        if kind == "free_body":
            ctx, n = build_free_body_replay(self._vis, record_dir)
            self._replay = ("free_body", ctx, n)
        else:
            state, period, model, env = build_kinematics_replay(
                self._vis, record_dir / "config.yaml",
                vehicle_path=record_dir / "vehicle.yaml")
            self._replay = ("kinematics", state, period, model, env, 120)
        self._frame = 0.0
        n_frames = self._replay[2] if self._replay[0] == "free_body" else self._replay[5]
        self.frame_slider.setRange(0, n_frames - 1)
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self._render_frame(0)

    def show_record_frame(self, record_dir, frame: int = 0) -> None:
        """Render one frame of a record (kept for single-frame callers)."""
        self.set_record(record_dir)
        self._render_frame(frame)

    # --- playback -------------------------------------------------------------

    def toggle_play(self) -> None:
        if self._replay is None:
            return
        self._playing = not self._playing
        self.play_button.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def stop(self) -> None:
        self._playing = False
        self._timer.stop()
        self.play_button.setText("Play")

    def _tick(self) -> None:
        """Advance the playback by one timer tick's worth of frames."""
        speed = float(self.speed_combo.currentText())
        if self._replay[0] == "free_body":
            _, ctx, n = self._replay
            duration = ctx.ts.t[-1] - ctx.ts.t[0]
            advance = (_TICK_MS / 1000.0) * speed * n / max(duration, 1e-9)
        else:
            _, _, period, _, _, frames_per_beat = self._replay
            advance = (_TICK_MS / 1000.0) * speed * frames_per_beat / period
        self._frame = (self._frame + advance) % self._n_frames()
        self._render_frame(int(self._frame))

    def _slider_released(self) -> None:
        self._frame = float(self.frame_slider.value())
        self._render_frame(int(self._frame))

    def seek_time(self, t: float) -> None:
        """Scrub the 3D view to the record time t (cursor linkage, M6 T3)."""
        if self._replay is None:
            return
        if self._replay[0] == "free_body":
            _, ctx, n = self._replay
            i = int(np.argmin(np.abs(ctx.ts.t - t)))
        else:
            _, _, period, _, _, frames_per_beat = self._replay
            i = int((t % period) / period * frames_per_beat)
        self._frame = float(i)
        self._render_frame(i)

    def _n_frames(self) -> int:
        return self._replay[2] if self._replay[0] == "free_body" else self._replay[5]

    def _render_frame(self, i: int) -> None:
        from fwmav_sim.viz.meshcat_scene import pose_frame, render_free_body_frame

        if self._replay is None:
            return
        if self._replay[0] == "free_body":
            _, ctx, n = self._replay
            render_free_body_frame(self._vis, ctx, i % n)
        else:
            _, state, period, model, env, frames_per_beat = self._replay
            pose_frame(self._vis, state, (i % frames_per_beat) / frames_per_beat * period,
                       model, env)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(i)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"{i + 1}/{self._n_frames()}")
