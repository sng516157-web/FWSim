"""M9 workbench shell (T3): the live-first window.

Layout (docs/design/M9_workbench_redesign.md §3.2):

    left:   library palette (mechanical primitives + control blocks)
    centre: tabs [platform | control | 3D live | scopes] + transport bar
    right:  properties of the selected node + dashboard (live tuning)

The transport bar drives a LiveSession (play/pause/stop/speed/record);
stop saves the run record — live is the default mode, recording always on.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QComboBox,
        QDockWidget,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QPushButton,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:  # pragma: no cover
    _HAVE_QT = False

# node catalog: (section, label, kind-tag)
PALETTE = [
    ("mechanical", "body: box", "body.box"),
    ("mechanical", "body: cylinder", "body.cylinder"),
    ("mechanical", "body: point mass", "body.point"),
    ("mechanical", "joint: revolute", "joint.revolute"),
    ("mechanical", "wing (draw outline)", "wing"),
    ("mechanical", "CAD subtree (URDF)", "cad"),
    ("control", "sensor: IMU", "sensor_imu"),
    ("control", "sensor: mocap", "sensor_mocap"),
    ("control", "estimator: complementary", "estimator_complementary"),
    ("control", "controller: PID", "controller_pid"),
    ("control", "controller: LQR", "controller_lqr"),
    ("control", "mixer: wingbeat channels", "mixer_wingbeat_channels"),
    ("control", "actuator: torque", "actuator_torque"),
    ("control", "scope", "scope"),
]


if _HAVE_QT:

    class DashboardPanel(QWidget):
        """Live-tuning sliders for the running pipeline's live params (M9 T4,
        Simulink Dashboard pattern): each tunable block param gets a slider +
        spinbox; moving it writes through to the running plant's pipeline via
        Pipeline.set_param. Nothing here exists when no live session runs."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._layout = QVBoxLayout(self)
            self._session = None
            self._rows: list[tuple[str, str, QSlider, QLabel, tuple[float, float]]] = []
            self._layout.addWidget(QLabel("no live session — press ▶"))
            self._layout.addStretch(1)

        def bind(self, session) -> None:
            """Populate sliders from the session's pipeline live params."""
            while self._layout.count():
                item = self._layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._rows = []
            self._session = session
            pipe = getattr(session.plant, "pipeline", None)
            if pipe is None:
                self._layout.addWidget(QLabel("experiment has no control pipeline"))
                self._layout.addStretch(1)
                return
            lp = pipe.live_params()
            if not lp:
                self._layout.addWidget(QLabel("no live-tunable params in this pipeline"))
                self._layout.addStretch(1)
                return
            for block_id, params in lp.items():
                self._layout.addWidget(QLabel(f"[{block_id}]"))
                for name, value in params.items():
                    row = QHBoxLayout()
                    lab = QLabel(name)
                    slider = QSlider(Qt.Horizontal)
                    lo, hi = self._range_for(name, value)
                    slider_min, slider_max = 0, 1000
                    slider.setRange(slider_min, slider_max)
                    slider.setValue(int(1000 * (value - lo) / (hi - lo)))
                    val = QLabel(f"{value:.4g}")
                    row.addWidget(lab)
                    row.addWidget(slider, 1)
                    row.addWidget(val)
                    self._layout.addLayout(row)
                    self._rows.append((block_id, name, slider, val, (lo, hi)))
                    slider.valueChanged.connect(self._make_handler(block_id, name, slider, val, (lo, hi)))
            self._layout.addStretch(1)

        @staticmethod
        def _range_for(name: str, value: float) -> tuple[float, float]:
            if name in ("kp", "ki", "kd"):
                return (-10 * abs(value) - 1e-3, 10 * abs(value) + 1e-3)
            if name == "setpoint":
                return (value - 1.0, value + 1.0)
            if name == "limit":
                return (0.0, max(2 * value, 1.0))
            return (0.0, max(2 * value, 1e-3))

        def _make_handler(self, block_id, name, slider, val_label, rng):
            def _on(pos: int):
                lo, hi = rng
                value = lo + (hi - lo) * pos / 1000.0
                val_label.setText(f"{value:.4g}")
                if self._session is not None:
                    self._session.plant.pipeline.set_param(block_id, name, value)
            return _on

    class TransportBar(QWidget):
        """play / pause / stop / speed / record — the Webots-style transport."""

        def __init__(self, parent=None):
            super().__init__(parent)
            lay = QHBoxLayout(self)
            lay.setContentsMargins(4, 2, 4, 2)
            self.play = QPushButton("▶ play")
            self.play.setCheckable(True)
            self.stop = QPushButton("⏹ stop + save")
            self.speed = QComboBox()
            self.speed.addItems(["0.05×", "0.1×", "0.25×", "0.5×", "1×"])
            self.speed.setCurrentIndex(1)
            self.record = QPushButton("● record")
            self.record.setCheckable(True)
            self.record.setChecked(True)  # recording is the default (M9)
            self.status = QLabel("idle")
            lay.addWidget(self.play)
            lay.addWidget(self.stop)
            lay.addWidget(QLabel("speed"))
            lay.addWidget(self.speed)
            lay.addWidget(self.record)
            lay.addStretch(1)
            lay.addWidget(self.status)

        def speed_value(self) -> float:
            return float(self.speed.currentText().rstrip("×"))

    class LibraryPalette(QListWidget):
        """The node catalog (Simulink library-browser role)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            for section, label, tag in PALETTE:
                self.addItem(f"{section}  ·  {label}")
                item = self.item(self.count() - 1)
                item.setData(Qt.UserRole, tag)


    class WorkbenchWindow(QMainWindow):
        """The M9 live-first workbench. Reuses the M5-M8 panels; the run/batch
        path stays available via the CLI and the legacy window."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("FWSim Workbench")
            self.resize(1400, 900)

            from fwmav_sim.ui.nodes.pipeline_canvas import PipelineEditorWidget
            from fwmav_sim.ui.nodes.platform_tree import PlatformTreeWidget
            from fwmav_sim.ui.scene3d import Scene3DPanel
            from fwmav_sim.ui.timeseries import TimeSeriesPanel

            # centre: tabs + transport
            self.tabs = QTabWidget()
            # platform tab: tree + wing planform editor (M9 T1/T2)
            from fwmav_sim.ui.nodes.planform_canvas import PlanformCanvas
            from fwmav_sim.ui.nodes.platform_tree import PlatformTreeWidget

            self.platform_tree = PlatformTreeWidget()
            self.planform_canvas = PlanformCanvas()
            platform_split = QWidget()
            pbox = QVBoxLayout(platform_split)
            pbox.addWidget(self.platform_tree, 1)
            pbox.addWidget(QLabel("wing outline (drag the points):"))
            pbox.addWidget(self.planform_canvas, 2)
            self.pipeline_editor = PipelineEditorWidget()
            # 3D: Qt-native GL scene (M9 T5); MeshCat panel kept for the
            # legacy window until parity is confirmed on real hardware
            from fwmav_sim.viz.gl_scene import GLSceneWidget

            self.scene3d = GLSceneWidget()
            self._scene3d_legacy = Scene3DPanel  # retained, not instantiated
            self.timeseries = TimeSeriesPanel()
            self.tabs.addTab(platform_split, "platform")
            self.tabs.addTab(self.pipeline_editor, "control")
            self.tabs.addTab(self.scene3d, "3D live")
            self.tabs.addTab(self.timeseries, "scopes")
            self.transport = TransportBar()
            centre = QWidget()
            vbox = QVBoxLayout(centre)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.addWidget(self.tabs, 1)
            vbox.addWidget(self.transport, 0)
            self.setCentralWidget(centre)

            # left: palette
            self.palette = LibraryPalette()
            dock_l = QDockWidget("library", self)
            dock_l.setWidget(self.palette)
            self.addDockWidget(Qt.LeftDockWidgetArea, dock_l)

            # right: properties + dashboard (M9 T4: live tuning sliders)
            from fwmav_sim.ui.parameter_editor import ParameterEditor
            from fwmav_sim.ui.validity_panel import ValidityPanel

            self.properties = ParameterEditor(lambda _u: None)
            self.validity = ValidityPanel()
            right = QTabWidget()
            right.addTab(self.properties, "properties")
            right.addTab(self.validity, "validity")
            self.dashboard = DashboardPanel()
            right.addTab(self.dashboard, "dashboard")
            dock_r = QDockWidget("node", self)
            dock_r.setWidget(right)
            self.addDockWidget(Qt.RightDockWidgetArea, dock_r)

            self.statusBar().showMessage(
                "double-click the library to add nodes; press ▶ to run live")

            # menu: open experiment / open record (record scrubs into the GL scene)
            menu = self.menuBar().addMenu("file")
            menu.addAction("open experiment…", self._open_experiment)
            menu.addAction("open record…", self._open_record)

            # wiring: palette double-click adds to the active editor
            self.palette.itemDoubleClicked.connect(self._on_palette_add)
            self.transport.play.toggled.connect(self._on_play)
            self.transport.stop.clicked.connect(self._on_stop)
            self._live_timer = None
            self._live_session = None
            self._experiment_path: Path | None = None

        # --- palette ---------------------------------------------------------

        def _on_palette_add(self, item) -> None:
            tag = item.data(Qt.UserRole)
            if self.tabs.currentWidget() is self.pipeline_editor:
                if tag in ("sensor_imu", "sensor_mocap", "estimator_complementary",
                           "controller_pid", "controller_lqr",
                           "mixer_wingbeat_channels", "actuator_torque", "scope"):
                    scene = self.pipeline_editor.canvas.scene()
                    # auto-id: type_N
                    n = sum(1 for b in scene.graph.blocks.values()
                            if b["type"] == tag) + 1
                    try:
                        scene.add_block(f"{tag}_{n}", tag)
                    except Exception as exc:  # noqa: BLE001
                        self.statusBar().showMessage(str(exc))
                else:
                    self.statusBar().showMessage(
                        f"{tag} is a mechanical node — switch to the platform tab")
            else:
                self.statusBar().showMessage(
                    f"{tag}: platform tree editing via the property panel "
                    "(full drag-add lands with the tree editor rework)")

        # --- live transport -----------------------------------------------------

        def _open_experiment(self) -> None:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(self, "open experiment",
                                                  "experiments", "YAML (*.yaml)")
            if path:
                self.load_experiment(Path(path))

        def _open_record(self) -> None:
            """Scrub a saved record into the GL scene (M9 T5 parity: replay)."""
            from PySide6.QtWidgets import QFileDialog

            path = QFileDialog.getExistingDirectory(self, "open run record", "outputs")
            if not path:
                return
            from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
            from fwmav_sim.environment.spec import environment_from_spec
            from fwmav_sim.experiment.config import (
                kinematics_map,
                load_experiment,
                load_vehicle,
            )
            from fwmav_sim.experiment.record import load_record
            from fwmav_sim.viz.gl_frames import frame_geometry
            from fwmav_sim.viz.meshcat_scene import _FreeBodyReplayCtx

            rd = Path(path)
            ts, _meta = load_record(rd)
            vehicle = load_vehicle(rd / "vehicle.yaml")
            exp = load_experiment(rd / "config.yaml")
            ctx = _FreeBodyReplayCtx(
                ts=ts, vehicle=vehicle, env=environment_from_spec(exp.environment),
                model=QuasiSteadyBladeElement(vehicle.aero),
                kin_map=kinematics_map(exp, vehicle), art=None, state=None,
                force_scale=5.0)
            meshes, arrows = frame_geometry(ctx, 0)
            self.scene3d.set_wing_meshes(meshes)
            self.scene3d.set_force_arrows(arrows)
            self.scene3d.update()
            self.timeseries.set_record(ts)
            self.tabs.setCurrentWidget(self.scene3d)
            self.statusBar().showMessage(f"record: {rd.name} (frame 0; scope cursor scrubs)")

        def load_experiment(self, path: Path) -> None:
            """The experiment the live transport runs."""
            self._experiment_path = Path(path)
            self.statusBar().showMessage(f"loaded {path.name} — press ▶")

        def _on_play(self, on: bool) -> None:
            if self._experiment_path is None:
                self.statusBar().showMessage("load an experiment first "
                                             "(fwmav-sim gui or file menu)")
                self.transport.play.setChecked(False)
                return

            if on:
                try:
                    from fwmav_sim.viz.gl_live import GLLiveSession

                    self._live_session = GLLiveSession(self._experiment_path,
                                                       self.scene3d)
                except (ValueError, ImportError) as exc:
                    self.statusBar().showMessage(f"live failed: {exc}")
                    self.transport.play.setChecked(False)
                    return
                # QTimer in the GUI thread: Qt widgets are not thread-safe,
                # so no worker thread touches the GL widget
                from PySide6.QtCore import QTimer

                speed = self.transport.speed_value()
                frame_wall = 1.0 / 30.0
                self._live_timer = QTimer(self)
                self._live_timer.setInterval(int(frame_wall * 1000))

                def tick():
                    try:
                        self._live_session.step_frame(frame_wall * speed)
                    except Exception as exc:  # noqa: BLE001
                        self._live_timer.stop()
                        self.statusBar().showMessage(f"live stopped: {exc}")

                self._live_timer.timeout.connect(tick)
                self._live_timer.start()
                self.dashboard.bind(self._live_session)
                self.transport.status.setText("live" +
                                              (" (recording)" if self.transport.record.isChecked()
                                               else ""))
            else:
                if self._live_timer is not None:
                    self._live_timer.stop()
                self.transport.status.setText("paused")

        def _on_stop(self) -> None:
            if self._live_timer is not None:
                self._live_timer.stop()
                self._live_timer = None
            self.transport.play.setChecked(False)
            if self._live_session is not None and self.transport.record.isChecked():
                rec = self._live_session.save_record()
                if rec is not None:
                    self.statusBar().showMessage(f"record saved: {rec}")
            self._live_session = None
            self.transport.status.setText("idle")
