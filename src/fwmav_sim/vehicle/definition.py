"""Vehicle assembly and validation (M2 Task 1).

Schema-level checks (positive mass, SPD inertia, unit joint axes) are enforced
by ``vehicle/schema.py`` at load time. This module adds assembly-level
validation: referential integrity of the joint tree, a single root body, and
explicit left/right wing symmetry (frames.md §4).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.vehicle.schema import VehicleDefinition


def validate_vehicle(vehicle: VehicleDefinition) -> None:
    """Assembly-level validation beyond the per-model schema checks."""
    bodies = {b.name for b in vehicle.bodies}
    if len(bodies) != len(vehicle.bodies):
        raise ValueError("duplicate body names")

    # joint tree: referential integrity, single root, no cycles
    parents = set()
    for j in vehicle.joints:
        if j.parent_body not in bodies or j.child_body not in bodies:
            raise ValueError(
                f"joint {j.name!r} references unknown body "
                f"({j.parent_body!r}, {j.child_body!r})"
            )
        if j.child_body in parents:
            raise ValueError(f"body {j.child_body!r} has two parent joints")
        parents.add(j.child_body)
    if vehicle.joints:
        roots = bodies - parents
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root body, got {sorted(roots)}")
        # cycle check via parent walk
        child_to_parent = {j.child_body: j.parent_body for j in vehicle.joints}
        for b in bodies:
            seen = set()
            cur = b
            while cur in child_to_parent:
                cur = child_to_parent[cur]
                if cur in seen:
                    raise ValueError("joint tree contains a cycle")
                seen.add(cur)

    # wings: sides must come in declared mirrored pairs (frames.md §4)
    wings = vehicle.wings
    sides = [w.side for w in wings]
    if sides.count("right") != sides.count("left"):
        raise ValueError("wings must come in left/right mirrored pairs")
    for wr in [w for w in wings if w.side == "right"]:
        candidates = [w for w in wings if w.side == "left"
                      and w.span.value == wr.span.value
                      and w.root_chord.value == wr.root_chord.value
                      and w.tip_chord.value == wr.tip_chord.value
                      and np.allclose(np.asarray(w.hinge_location.value),
                                      np.asarray(wr.hinge_location.value) * [1, -1, 1])]
        if not candidates:
            raise ValueError(f"right wing {wr.name!r} has no mirrored left partner")


def joint_by_name(vehicle: VehicleDefinition, name: str):
    for j in vehicle.joints:
        if j.name == name:
            return j
    raise KeyError(name)


def body_by_name(vehicle: VehicleDefinition, name: str):
    for b in vehicle.bodies:
        if b.name == name:
            return b
    raise KeyError(name)
