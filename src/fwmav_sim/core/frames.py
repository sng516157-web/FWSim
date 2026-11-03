"""Registry of the six coordinate frames frozen in docs/conventions/frames.md.

The frame chain is W <- B <- SP <- WR <- BE, with WR instantiated per wing
side (WR right / WR left). This registry is the machine-readable mirror of
the convention document; the two must stay in sync (test_frames_registry.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    """A named coordinate frame and its place in the transform chain."""

    name: str
    parent: str | None
    description: str


#: All frames defined by docs/conventions/frames.md, keyed by name.
#: ``parent: None`` marks the root (world) frame.
FRAMES: dict[str, Frame] = {
    "W": Frame(
        name="W",
        parent=None,
        description="World frame, NED: x north, y east, z down; gravity g = [0, 0, +9.81] m/s^2.",
    ),
    "B": Frame(
        name="B",
        parent="W",
        description="Body (thorax) frame: x forward, y right, z down; coincides with W at zero attitude.",
    ),
    "SP": Frame(
        name="SP",
        parent="B",
        description="Stroke-plane frame: B rotated by stroke-plane angle beta about the B y-axis.",
    ),
    "WR_R": Frame(
        name="WR_R",
        parent="SP",
        description="Right wing-root frame: span along +y, chord along x (Sane & Dickinson convention).",
    ),
    "WR_L": Frame(
        name="WR_L",
        parent="SP",
        description="Left wing-root frame: right-handed mirror of WR_R via reflection S = diag(1, -1, 1); "
        "span along -y_SP, chord along +x, z-axis opposite to the physical wing normal.",
    ),
    "BE": Frame(
        name="BE",
        parent="WR_R",
        description="Blade-element frame: origin at the element centre, axes aligned with the parent WR.",
    ),
}

#: The documented transform chain, parent <- child, for the right wing.
#: For the left wing, ``WR_L`` takes the place of ``WR_R``.
TRANSFORM_CHAIN: tuple[str, ...] = ("W", "B", "SP", "WR_R", "BE")
