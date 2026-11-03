"""Unit tests for the frame registry (M0 stage file §0.4).

The registry in core/frames.py must list all six frames with parent/child
relationships matching docs/conventions/frames.md.
"""

from __future__ import annotations

from fwmav_sim.core.frames import FRAMES, TRANSFORM_CHAIN

EXPECTED_FRAMES = {"W", "B", "SP", "WR_R", "WR_L", "BE"}
EXPECTED_PARENTS = {
    "W": None,
    "B": "W",
    "SP": "B",
    "WR_R": "SP",
    "WR_L": "SP",
    "BE": "WR_R",
}


def test_registry_lists_all_six_frames():
    assert set(FRAMES) == EXPECTED_FRAMES


def test_parent_child_relationships_match_frames_doc():
    for name, parent in EXPECTED_PARENTS.items():
        assert FRAMES[name].parent == parent, name


def test_single_root_frame():
    roots = [f.name for f in FRAMES.values() if f.parent is None]
    assert roots == ["W"]


def test_transform_chain_is_contiguous():
    chain = list(TRANSFORM_CHAIN)
    assert chain[0] == "W"
    for child, parent in zip(chain[1:], chain[:-1], strict=True):
        assert FRAMES[child].parent == parent
