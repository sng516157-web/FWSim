"""MeshCat 3D scene for the fixed-body slice (M1 Task 10).

Live animated scene: each wing as a discretized surface posed by the frozen
transform chain, blade-element frame triads, per-element force arrows (from
the aero model), and a CoM marker. Geometry and motion always come from the
vehicle definition + prescribed kinematics; forces from the model — nothing
hard-coded (brief §8.2).

Run a live view of an experiment:

    fwmav-sim view experiments/m1_fixed_body_hover.yaml

or replay a stored run record:

    fwmav-sim view outputs/<run_dir>

Then open the printed URL (default http://127.0.0.1:7000/static/) in a browser.
Verified on Windows in M1 (server start + geometry push; browser rendering
checked by the user).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.core.transforms import R_B_SP, R_W_B
from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.kinematics.blade_element import (
    chord_at,
    discretize,
    element_state,
    wing_chain,
)
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics, wing_state
from fwmav_sim.vehicle.schema import WingSpec


def _wing_mesh_vertices(wing: WingSpec, edges: np.ndarray) -> np.ndarray:
    """Wing surface quad-strip vertices in WR coords (pitch axis honoured).

    +x_WR points toward the leading edge, +y_WR is the span (outboard on the
    wing's own side) for BOTH wings — left/right mirroring lives entirely in
    the frozen R_SP_WR_L chain (frames.md §4), never in the geometry.
    The pitch axis is at pitch_axis_fraction of the chord from the LE.
    """
    x0 = wing.pitch_axis_fraction.value
    verts = []
    for s in edges:
        c = chord_at(wing, s)
        verts.append([(x0 - 1.0) * c, s, 0.0])  # trailing-edge side
        verts.append([x0 * c, s, 0.0])  # leading-edge side
    return np.asarray(verts)


def build_scene(
    vis,
    wings: list[WingSpec],
    kinematics: PrescribedWingKinematics,
    n_elements: int = 16,
    force_scale: float = 1.0,
) -> dict:
    """Create the static scene content. Returns handles updated per frame."""
    import meshcat.geometry as g

    # meshcat's default helpers assume Z-up; our world is NED (z down,
    # frames.md §1.1), so hide them to avoid sign confusion
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)

    state = {"wings": []}
    for wing in wings:
        elements = discretize(wing, n_elements)
        edges = np.linspace(0.0, wing.span.value, n_elements + 1)
        verts = _wing_mesh_vertices(wing, edges)
        faces = []
        for k in range(n_elements):
            b = 2 * k
            faces += [[b, b + 1, b + 3], [b, b + 3, b + 2]]
        mesh = g.TriangularMeshGeometry(verts, np.asarray(faces))
        vis[f"wing/{wing.name}"].set_object(
            mesh, g.MeshLambertMaterial(color=0x44AAFF, opacity=0.85,
                                        transparent=True, doubleSided=True)
        )
        vis[f"frames/{wing.name}_root"].set_object(g.triad(scale=0.03))
        vis[f"frames/{wing.name}_tip"].set_object(g.triad(scale=0.02))
        vis[f"forces/{wing.name}"].set_object(
            g.LineSegments(g.PointsGeometry(np.zeros((2 * n_elements, 3))),
                           g.LineBasicMaterial(color=0xFF5533))
        )
        state["wings"].append({"spec": wing, "elements": elements})
    vis["com"].set_object(g.Sphere(0.004), g.MeshLambertMaterial(color=0xFF3333))
    vis["frames/world"].set_object(g.triad(scale=0.05))
    state["force_scale"] = force_scale
    return state


def pose_frame(vis, state: dict, t: float, model=None, env=None) -> None:
    """Pose all wings and force arrows at time t (one animation frame)."""
    import meshcat.geometry as g

    kin = state["kinematics"]
    beta = kin.stroke_plane_angle.value
    for entry in state["wings"]:
        wing, elements = entry["spec"], entry["elements"]
        ks = wing_state(kin, t)
        R_WR, _, _, _ = wing_chain(ks, wing.side)
        R_world_WR = R_W_B(0.0, 0.0, 0.0) @ R_B_SP(beta) @ R_WR
        origin = np.asarray(wing.hinge_location.value, dtype=float)

        tf = np.eye(4)
        tf[:3, :3] = R_world_WR
        tf[:3, 3] = origin
        vis[f"wing/{wing.name}"].set_transform(tf)
        vis[f"frames/{wing.name}_root"].set_transform(tf)

        tf_tip = tf.copy()
        tip = np.array([elements[-1].x_c, elements[-1].s, 0.0])
        tf_tip[:3, 3] = origin + R_world_WR @ tip
        vis[f"frames/{wing.name}_tip"].set_transform(tf_tip)

        if model is not None and env is not None:
            seg = np.zeros((2 * len(elements), 3))
            for j, el in enumerate(elements):
                st = element_state(wing, el, kin, t, nu=env.nu, v_air_W=env.v_air_W)
                f = model.element_forces(st, env).resultant
                seg[2 * j] = st.r_W
                seg[2 * j + 1] = st.r_W + state["force_scale"] * f
            vis[f"forces/{wing.name}"].set_object(
                g.LineSegments(g.PointsGeometry(seg), g.LineBasicMaterial(color=0xFF5533))
            )


def serve_live(experiment_path: Path, playback_speed: float = 0.05, force_scale: float = 5.0) -> str:
    """Start MeshCat and animate the wingbeat in a loop. Blocks (Ctrl-C stops).

    playback_speed < 1 slows the wingbeat down for viewing.
    force_scale is the arrow display scale in metres per newton (display-only).
    Returns the URL (also printed).
    """
    import meshcat

    exp = load_experiment(experiment_path)
    vehicle = load_vehicle(Path(experiment_path).parent / exp.vehicle)
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)

    vis = meshcat.Visualizer()
    state = build_scene(vis, vehicle.wings, exp.kinematics,
                        n_elements=vehicle.blade_elements.n_elements,
                        force_scale=force_scale)
    state["kinematics"] = exp.kinematics
    url = vis.url()
    print(f"MeshCat live view: {url}   (Ctrl-C to stop)")

    period = 1.0 / exp.kinematics.flap_frequency.value
    frames_per_beat = 120
    try:
        while True:
            for k in range(frames_per_beat):
                pose_frame(vis, state, k / frames_per_beat * period, model, env)
                time.sleep(period * playback_speed / frames_per_beat)
    except KeyboardInterrupt:
        pass
    return url


def replay_record(record_dir: Path, playback_speed: float = 0.05, force_scale: float = 5.0) -> str:
    """Replay a stored run record live (forces from the record's config)."""
    record_dir = Path(record_dir)
    return serve_live(record_dir / "config.yaml", playback_speed, force_scale)


def spike_check() -> bool:
    """Install/runtime check: start a MeshCat server and push geometry.

    Returns True if the server accepted commands without error. Browser
    rendering itself cannot be verified headlessly; the URL is printed for a
    human check (recorded in the M1 report).
    """
    try:
        import meshcat
        import meshcat.geometry as g

        vis = meshcat.Visualizer()
        verts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        faces = np.array([[0, 1, 2]])
        vis["spike"]["wing"].set_object(
            g.Mesh(g.TriangularMeshGeometry(verts, faces), g.MeshLambertMaterial(color=0x44AAFF))
        )
        vis["spike"]["com"].set_object(g.Sphere(0.005), g.MeshLambertMaterial(color=0xFF3333))
        vis["spike"]["frames"].set_object(g.triad(scale=0.05))
        print(f"MeshCat server OK: {vis.url()}")
        return True
    except Exception as exc:  # noqa: BLE001 - spike must record any failure
        print(f"MeshCat spike FAILED: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        serve_live(Path(sys.argv[1]))
    else:
        sys.exit(0 if spike_check() else 1)
