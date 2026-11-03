"""MeshCat 3D scene for the fixed-body slice (M1 Task 10).

Minimal scene: each wing as a discretized surface (blade-element quads),
blade-element frame triads, per-element force arrows, and a CoM marker.
The scene is built from a run record's stored arrays — never from hard-coded
geometry (brief §8.2).

Usage (manual spike):  python -m fwmav_sim.viz.meshcat_scene outputs/<run_dir>
opens a MeshCat server and prints the URL; the browser does the rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def wing_surface_mesh(wing_name: str, ts_meta: dict, vehicle_wing, elements) -> tuple:
    """Vertices/faces of the discretized wing surface at theta_w = 0 in SP.

    Placeholder for the full scene builder: geometry comes from the WingSpec +
    blade elements, posed at mid-stroke. Returns (vertices, faces).
    """
    R = vehicle_wing.span.value
    verts, faces = [], []
    sign = 1.0 if vehicle_wing.side == "right" else -1.0
    edges = np.linspace(0.0, R, len(elements) + 1)
    from fwmav_sim.kinematics.blade_element import chord_at

    for a, b in zip(edges[:-1], edges[1:], strict=True):
        ca, cb = chord_at(vehicle_wing, a), chord_at(vehicle_wing, b)
        base = len(verts)
        # quad: (LE/TE) x (root/tip edge), span along +y (right) / -y (left)
        verts += [[ca / 2, sign * a, 0.0], [-ca / 2, sign * a, 0.0],
                  [cb / 2, sign * b, 0.0], [-cb / 2, sign * b, 0.0]]
        faces += [[base, base + 1, base + 3], [base, base + 3, base + 2]]
    return np.asarray(verts), np.asarray(faces)


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
    ok = spike_check()
    if len(sys.argv) > 1:
        print(f"(scene build from record {Path(sys.argv[1])} — full scene wiring lands with M2)")
    sys.exit(0 if ok else 1)
