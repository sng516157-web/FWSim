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
from fwmav_sim.core.transforms import R_W_B, R_B_SP_wing, quat_to_matrix
from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.experiment.config import load_experiment, load_vehicle
from fwmav_sim.kinematics.blade_element import (
    chord_at,
    discretize,
    element_state,
    le_offset_at,
    wing_chain,
)
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics, wing_state
from fwmav_sim.vehicle.schema import WingSpec

#: Display-only rotation from our physics world (NED, z down, frames.md §1.1)
#: to the meshcat/three.js viewer convention (Y up). Physical up (-z) renders
#: as viewer +y; physical forward (+x) stays +x. det = +1 (proper rotation).
#: Physics is never affected — this mapping is applied only at the view layer.
NED_TO_VIEW = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _wing_mesh_vertices(wing: WingSpec, edges: np.ndarray) -> np.ndarray:
    """Wing surface quad-strip vertices in WR coords (pitch axis honoured).

    +x_WR points toward the leading edge, +y_WR is the span (outboard on the
    wing's own side) for BOTH wings — left/right mirroring lives entirely in
    the frozen R_SP_WR_L chain (frames.md §4), never in the geometry.
    The leading edge sits at le_offset_at(wing, s): pitch_axis_fraction * c
    for the analytic planforms, the swept outline for custom ones.
    """
    verts = []
    for s in edges:
        c = chord_at(wing, s)
        le = le_offset_at(wing, s)  # = f_pa * c for analytic planforms
        verts.append([le - c, s, 0.0])  # trailing-edge side
        verts.append([le, s, 0.0])  # leading-edge side
    return np.asarray(verts)


def _align_axis_to(d: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping the unit vector `axis` onto the unit vector d."""
    d = d / np.linalg.norm(d)
    k = np.cross(axis, d)
    c = float(axis @ d)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg about any axis perpendicular to `axis`
        perp = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-9:
            perp = np.cross(axis, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + kx + kx @ kx * (1.0 / (1.0 + c))


_Y = np.array([0.0, 1.0, 0.0])  # meshcat Cylinder axis convention (three.js)


def build_scene(
    vis,
    wings: list[WingSpec],
    kinematics: PrescribedWingKinematics,
    n_elements: int = 16,
    force_scale: float = 5.0,
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
        # one shaft + head per element; unit-height z-aligned cylinders
        # transformed per frame (display-only arrow scale)
        for j in range(n_elements):
            vis[f"forces/{wing.name}/{j}/shaft"].set_object(
                g.Cylinder(1.0, 0.0008), g.MeshLambertMaterial(color=0xFF2222))
            vis[f"forces/{wing.name}/{j}/head"].set_object(
                g.Cylinder(1.0, radiusTop=0.0, radiusBottom=0.0025),
                g.MeshLambertMaterial(color=0xFF2222))
        state["wings"].append({"spec": wing, "elements": elements})
    vis["com"].set_object(g.Sphere(0.004), g.MeshLambertMaterial(color=0xFF3333))
    vis["frames/world"].set_object(g.triad(scale=0.05))
    state["force_scale"] = force_scale
    return state


def pose_frame(vis, state: dict, t: float, model=None, env=None) -> None:
    """Pose all wings and force arrows at time t (one animation frame)."""
    kin = state["kinematics"]
    beta = kin.stroke_plane_angle.value
    for entry in state["wings"]:
        wing, elements = entry["spec"], entry["elements"]
        ks = wing_state(kin, t)
        R_WR, _, _, _ = wing_chain(ks, wing.side)
        from fwmav_sim.kinematics.blade_element import dihedral_of
        R_world_WR = (R_W_B(0.0, 0.0, 0.0)
                      @ R_B_SP_wing(beta, dihedral_of(wing), wing.side) @ R_WR)
        origin = NED_TO_VIEW @ np.asarray(wing.hinge_location.value, dtype=float)

        tf = np.eye(4)
        tf[:3, :3] = NED_TO_VIEW @ R_world_WR
        tf[:3, 3] = origin
        vis[f"wing/{wing.name}"].set_transform(tf)
        vis[f"frames/{wing.name}_root"].set_transform(tf)

        tf_tip = tf.copy()
        tip = np.array([elements[-1].x_c, elements[-1].s, 0.0])
        tf_tip[:3, 3] = origin + NED_TO_VIEW @ R_world_WR @ tip
        vis[f"frames/{wing.name}_tip"].set_transform(tf_tip)

        if model is not None and env is not None:
            for j, el in enumerate(elements):
                st = element_state(wing, el, kin, t, nu=env.nu, v_air_W=env.v_air_W)
                f = model.element_forces(st, env).resultant
                L = float(np.linalg.norm(f)) * state["force_scale"]
                shaft = vis[f"forces/{wing.name}/{j}/shaft"]
                head = vis[f"forces/{wing.name}/{j}/head"]
                if L < 1e-4:
                    shaft.set_transform(np.zeros((4, 4)))
                    head.set_transform(np.zeros((4, 4)))
                    continue
                r_view = NED_TO_VIEW @ st.r_W
                d = NED_TO_VIEW @ (f / np.linalg.norm(f))
                R_al = _align_axis_to(d, _Y)  # cylinder axis is +y (three.js convention)
                L_shaft, L_head = 0.8 * L, 0.2 * L
                tf_s = np.eye(4)
                tf_s[:3, :3] = R_al @ np.diag([1.0, L_shaft, 1.0])
                tf_s[:3, 3] = r_view + d * (L_shaft / 2)
                shaft.set_transform(tf_s)
                tf_h = np.eye(4)
                tf_h[:3, :3] = R_al @ np.diag([1.0, L_head, 1.0])
                tf_h[:3, 3] = r_view + d * (L_shaft + L_head / 2)
                head.set_transform(tf_h)


def build_kinematics_replay(vis, config_path: Path, force_scale: float = 5.0,
                            vehicle_path: Path | None = None):
    """Scene state for animating a fixed-body record / experiment config on an
    existing MeshCat server. Returns (state, period, model, env). Frame i of n
    is rendered with pose_frame(vis, state, i/n*period, model, env).

    vehicle_path overrides the vehicle file resolution; needed when the config
    is a run-record copy (its vehicle path is relative to the ORIGINAL
    experiment location, not the record dir — pass record_dir/vehicle.yaml).
    """
    exp = load_experiment(config_path)
    vehicle = load_vehicle(vehicle_path or (Path(config_path).parent / exp.vehicle))
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    state = build_scene(vis, vehicle.wings, exp.kinematics,
                        n_elements=vehicle.blade_elements.n_elements,
                        force_scale=force_scale)
    state["kinematics"] = exp.kinematics
    return state, 1.0 / exp.kinematics.flap_frequency.value, model, env


def serve_live(
    experiment_path: Path, playback_speed: float = 0.05, force_scale: float = 5.0,
    vehicle_path: Path | None = None,
) -> str:
    """Start MeshCat and animate the wingbeat in a loop. Blocks (Ctrl-C stops).

    playback_speed < 1 slows the wingbeat down for viewing.
    force_scale is the arrow display scale in metres per newton (display-only).
    vehicle_path: override vehicle resolution (run-record configs; see
    build_kinematics_replay).
    Returns the URL (also printed).
    """
    import meshcat

    vis = meshcat.Visualizer()
    state, period, model, env = build_kinematics_replay(vis, experiment_path,
                                                        force_scale, vehicle_path)
    url = vis.url()
    print(f"MeshCat live view: {url}   (Ctrl-C to stop)")

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
    """Replay a stored run record live. Dispatches on record kind: fixed-body
    records replay the kinematics; free_body records replay the actual
    recorded trajectory (body translates/rotates under the flapping wings)."""
    record_dir = Path(record_dir)
    import json

    meta = json.loads(record_dir.joinpath("metadata.json").read_text(encoding="utf-8"))
    kind = meta.get("time_series_meta", {}).get("kind")
    if kind == "free_body":
        return serve_free_body_record(record_dir, playback_speed, force_scale)
    return serve_live(record_dir / "config.yaml", playback_speed, force_scale,
                      vehicle_path=record_dir / "vehicle.yaml")


def build_free_body_scene(vis, ts, vehicle, exp, force_scale: float = 5.0):
    """Build the free-body scene from IN-MEMORY objects (record ts + vehicle +
    experiment). Frame i renders with render_free_body_frame(vis, ctx, i);
    live mode swaps ctx.ts per frame."""
    import meshcat.geometry as g

    from fwmav_sim.experiment.config import kinematics_map

    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    kin_map = kinematics_map(exp, vehicle)
    state = build_scene(vis, vehicle.wings, next(iter(kin_map.values())),
                        n_elements=vehicle.blade_elements.n_elements,
                        force_scale=force_scale)
    vis["body/thorax"].set_object(g.Box([0.04, 0.03, 0.02]),
                                  g.MeshLambertMaterial(color=0x888899))
    vis["body/abdomen"].set_object(g.Cylinder(0.03, 0.004),
                                   g.MeshLambertMaterial(color=0xBB7733))
    vis["body/com"].set_object(g.Sphere(0.004), g.MeshLambertMaterial(color=0x22FF66))

    from fwmav_sim.dynamics.spatial import ArticulatedModel

    ctx = _FreeBodyReplayCtx(ts=ts, vehicle=vehicle, env=env, model=model,
                             kin_map=kin_map, art=ArticulatedModel(vehicle),
                             state=state, force_scale=force_scale)
    return ctx


def build_free_body_replay(vis, record_dir: Path, force_scale: float = 5.0):
    """Replay context for a free_body record on an existing MeshCat server.
    Returns (ctx, n_frames); frame i renders with render_free_body_frame."""
    from fwmav_sim.experiment.record import load_record

    record_dir = Path(record_dir)
    ts, meta = load_record(record_dir)
    vehicle = load_vehicle(record_dir / "vehicle.yaml")
    exp = load_experiment(record_dir / "config.yaml")
    return build_free_body_scene(vis, ts, vehicle, exp, force_scale), len(ts.t)


def serve_free_body_record(record_dir: Path, playback_speed: float = 0.05,
                           force_scale: float = 5.0) -> str:
    """Live replay of a free_body run record: the recorded 6-DoF trajectory
    drives the body pose; wings follow the prescribed kinematics relative to
    the moving body; force arrows recomputed per element from the recorded
    state (position, velocity, attitude) at each frame."""
    import meshcat

    vis = meshcat.Visualizer()
    ctx, n = build_free_body_replay(vis, record_dir, force_scale)
    url = vis.url()
    print(f"MeshCat free-body replay: {url}   (Ctrl-C to stop)")

    t_rec = ctx.ts.t
    period_total = t_rec[-1] - t_rec[0]
    try:
        while True:
            for i in range(n):
                render_free_body_frame(vis, ctx, i)
                time.sleep(max(period_total / n / max(playback_speed, 1e-3), 0.005))
    except KeyboardInterrupt:
        pass
    return url


class _FreeBodyReplayCtx:
    """Everything needed to render one recorded free-body frame."""

    __slots__ = ("ts", "vehicle", "env", "model", "kin_map", "art", "state",
                 "force_scale", "joint")

    def __init__(self, ts, vehicle, env, model, kin_map, art, state, force_scale):
        self.ts = ts
        self.vehicle = vehicle
        self.env = env
        self.model = model
        self.kin_map = kin_map
        self.art = art
        self.state = state
        self.force_scale = force_scale
        self.joint = vehicle.joints[0] if vehicle.joints else None


def render_free_body_frame(vis, ctx: _FreeBodyReplayCtx, i: int) -> None:
    """Render recorded frame i: body pose, wings, CoM, abdomen, force arrows."""
    from fwmav_sim.kinematics.blade_element import WingFrameContext, dihedral_of, element_state_ctx

    ts = ctx.ts
    t_rec = ts.t
    P = ts["p"]
    Q = ts["quat"]
    QJ = ts["q_joints"]
    VW = ts["v_W"]
    OB = ts["omega_B"]

    R_wb = quat_to_matrix(Q[i])
    p_W = P[i]
    R_v = NED_TO_VIEW @ R_wb
    tf_body = np.eye(4)
    tf_body[:3, :3] = R_v
    tf_body[:3, 3] = NED_TO_VIEW @ p_W
    vis["body/thorax"].set_transform(tf_body)

    qm = np.concatenate([p_W, Q[i], QJ[i]])
    if ctx.joint is not None:
        tf_abd = ctx.art.forward_kinematics(qm)["abdomen"]
        r_J_W = p_W + R_wb @ np.asarray(ctx.joint.location.value)
        tf_a = np.eye(4)
        tf_a[:3, :3] = NED_TO_VIEW @ tf_abd[0] @ np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        tf_a[:3, 3] = NED_TO_VIEW @ (r_J_W + tf_abd[0] @ np.array([-0.015, 0.0, 0.0]))
        vis["body/abdomen"].set_transform(tf_a)

    r_com = ctx.art.com(qm)
    tf_c = np.eye(4)
    tf_c[:3, 3] = NED_TO_VIEW @ r_com
    vis["body/com"].set_transform(tf_c)

    for entry in ctx.state["wings"]:
        wing = entry["spec"]
        kin = ctx.kin_map[wing.name]
        ks = wing_state(kin, t_rec[i])
        R_WR, _, _, _ = wing_chain(ks, wing.side)
        R_world_WR = (R_wb @ R_B_SP_wing(kin.stroke_plane_angle.value,
                      dihedral_of(wing), wing.side) @ R_WR)
        origin_W = p_W + R_wb @ np.asarray(wing.hinge_location.value, float)
        tf_w = np.eye(4)
        tf_w[:3, :3] = NED_TO_VIEW @ R_world_WR
        tf_w[:3, 3] = NED_TO_VIEW @ origin_W
        vis[f"wing/{wing.name}"].set_transform(tf_w)
        vis[f"frames/{wing.name}_root"].set_transform(tf_w)
        tip = np.array([entry["elements"][-1].x_c, entry["elements"][-1].s, 0.0])
        tf_tip = tf_w.copy()
        tf_tip[:3, 3] = NED_TO_VIEW @ (origin_W + R_world_WR @ tip)
        vis[f"frames/{wing.name}_tip"].set_transform(tf_tip)

        ctx2 = WingFrameContext(wing, kin, t_rec[i], R_wb, p_W,
                                R_wb @ OB[i], VW[i], ctx.env.v_air_W)
        for j, el in enumerate(entry["elements"]):
            st = element_state_ctx(wing, el, ctx2, nu=ctx.env.nu)
            f = ctx.model.element_forces(st, ctx.env).resultant
            L = float(np.linalg.norm(f)) * ctx.force_scale
            shaft = vis[f"forces/{wing.name}/{j}/shaft"]
            head = vis[f"forces/{wing.name}/{j}/head"]
            if L < 1e-4:
                shaft.set_transform(np.zeros((4, 4)))
                head.set_transform(np.zeros((4, 4)))
                continue
            r_view = NED_TO_VIEW @ st.r_W
            d = NED_TO_VIEW @ (f / np.linalg.norm(f))
            R_al = _align_axis_to(d, _Y)
            L_shaft, L_head = 0.8 * L, 0.2 * L
            tf_s = np.eye(4)
            tf_s[:3, :3] = R_al @ np.diag([1.0, L_shaft, 1.0])
            tf_s[:3, 3] = r_view + d * (L_shaft / 2)
            shaft.set_transform(tf_s)
            tf_h = np.eye(4)
            tf_h[:3, :3] = R_al @ np.diag([1.0, L_head, 1.0])
            tf_h[:3, 3] = r_view + d * (L_shaft + L_head / 2)
            head.set_transform(tf_h)


def render_record_frame(vis, record_dir, frame: int = 0, force_scale: float = 5.0) -> None:
    """Render one frame of a free-body record on an existing MeshCat server
    (used by the M5 UI 3D panel)."""
    import meshcat.geometry as g

    from fwmav_sim.experiment.record import load_record

    ts, _meta = load_record(record_dir)
    vehicle = load_vehicle(Path(record_dir) / "vehicle.yaml")
    from fwmav_sim.experiment.config import kinematics_map, load_experiment

    exp = load_experiment(Path(record_dir) / "config.yaml")
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    kin_map = kinematics_map(exp, vehicle)
    state = build_scene(vis, vehicle.wings, next(iter(kin_map.values())),
                        n_elements=vehicle.blade_elements.n_elements,
                        force_scale=force_scale)
    vis["body/thorax"].set_object(g.Box([0.04, 0.03, 0.02]),
                                  g.MeshLambertMaterial(color=0x888899))
    if vehicle.joints:
        vis["body/abdomen"].set_object(g.Cylinder(0.03, 0.004),
                                       g.MeshLambertMaterial(color=0xBB7733))
    vis["body/com"].set_object(g.Sphere(0.004), g.MeshLambertMaterial(color=0x22FF66))
    from fwmav_sim.dynamics.spatial import ArticulatedModel

    ctx = _FreeBodyReplayCtx(ts=ts, vehicle=vehicle, env=env, model=model,
                             kin_map=kin_map, art=ArticulatedModel(vehicle),
                             state=state, force_scale=force_scale)
    render_free_body_frame(vis, ctx, frame % len(ts.t))


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
