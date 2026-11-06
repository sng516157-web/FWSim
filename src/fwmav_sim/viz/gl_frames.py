"""Frame adapter for the GL scene (M9 T5): computes per-frame wing poses and
force arrows in VIEW coords from a free-body record — the SAME math as
viz/meshcat_scene.render_free_body_frame, but returning geometry instead of
touching a MeshCat server. One computation, two renderers.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import R_B_SP_wing, quat_to_matrix
from fwmav_sim.kinematics.blade_element import (
    WingFrameContext,
    dihedral_of,
    element_state_ctx,
    le_offset_at,
    chord_at,
)
from fwmav_sim.kinematics.prescribed import wing_state
from fwmav_sim.kinematics.blade_element import wing_chain
from fwmav_sim.viz.meshcat_scene import NED_TO_VIEW


def wing_mesh_view(wing, R_world_WR, origin_W, n_elements: int = 16):
    """Wing quad-strip in VIEW coords at the given pose (custom planforms
    honoured via le_offset_at/chord_at)."""
    edges = np.linspace(0.0, wing.span.value, n_elements + 1)
    verts = []
    for s in edges:
        c = chord_at(wing, s)
        le = le_offset_at(wing, s)
        verts.append([le - c, s, 0.0])
        verts.append([le, s, 0.0])
    verts = np.asarray(verts)
    verts_W = (R_world_WR @ verts.T).T + origin_W
    faces = []
    for k in range(n_elements):
        b = 2 * k
        faces += [[b, b + 1, b + 3], [b, b + 3, b + 2]]
    from fwmav_sim.viz.meshcat_scene import NED_TO_VIEW as M

    return (M @ verts_W.T).T, np.asarray(faces)


def frame_geometry(ctx, i: int):
    """(wing_meshes, arrows) in VIEW coords for record frame i.

    ctx: a meshcat_scene._FreeBodyReplayCtx (ts, vehicle, env, model,
    kin_map, force_scale)."""
    ts = ctx.ts
    t_rec = ts.t
    P, Q, VW, OB = ts["p"], ts["quat"], ts["v_W"], ts["omega_B"]
    R_wb = quat_to_matrix(Q[i])
    p_W = P[i]

    meshes, arrows = [], []
    for wing in ctx.vehicle.wings:
        kin = ctx.kin_map[wing.name]
        ks = wing_state(kin, t_rec[i])
        R_WR, _, _, _ = wing_chain(ks, wing.side)
        R_world_WR = R_wb @ R_B_SP_wing(kin.stroke_plane_angle.value,
                                        dihedral_of(wing), wing.side) @ R_WR
        origin_W = p_W + R_wb @ np.asarray(wing.hinge_location.value, float)
        meshes.append(wing_mesh_view(wing, R_world_WR, origin_W,
                                     ctx.vehicle.blade_elements.n_elements))

        ctx2 = WingFrameContext(wing, kin, t_rec[i], R_wb, p_W,
                                R_wb @ OB[i], VW[i], ctx.env.v_air_W)
        from fwmav_sim.kinematics.blade_element import discretize

        for el in discretize(wing, ctx.vehicle.blade_elements.n_elements):
            st = element_state_ctx(wing, el, ctx2, nu=ctx.env.nu)
            f = ctx.model.element_forces(st, ctx.env).resultant
            L = float(np.linalg.norm(f)) * ctx.force_scale
            if L < 1e-4:
                continue
            arrows.append((NED_TO_VIEW @ st.r_W, NED_TO_VIEW @ (f / np.linalg.norm(f) * L)))
    return meshes, arrows
