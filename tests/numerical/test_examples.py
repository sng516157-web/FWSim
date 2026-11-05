"""Examples smoke test (M6.5 Task 2): the teaching scripts must keep working
with the current API — they run in CI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run_example(name: str, capsys):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    out = capsys.readouterr().out
    assert out, f"{name} printed nothing"
    return out


def test_fixed_body_example(capsys):
    out = _run_example("fixed_body_hover.py", capsys)
    assert "cycle-mean force" in out
    assert "wing_right" in out and "wing_left" in out


@pytest.mark.slow
def test_free_body_example(capsys):
    out = _run_example("free_body_hover.py", capsys)
    assert "status: OK" in out
    assert "peak sink rate" in out


@pytest.mark.slow
def test_closed_loop_example(capsys):
    out = _run_example("closed_loop_hover.py", capsys)
    assert "trim solution" in out
    assert "max |du_thrust|" in out
