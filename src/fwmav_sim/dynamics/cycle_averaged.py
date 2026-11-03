"""Cycle-averaged wing-force injection (M2 Task 7).

The reduced model consumes wingbeat-averaged forces **derived from an M1
time-resolved run**, loaded as data (brief §3.3) — never recomputed from an
independent formula. This module:

- computes per-wing cycle-averaged resultant forces from a fixed-body run
  record and saves them as a small JSON file (`fwmav-sim average-forces`),
- loads such a file for the reduced model, validating that the averaged
  file's vehicle/kinematics match the requester (mismatch => warning recorded
  in the run record, not silent acceptance).

Application point convention (documented approximation for the reduced
model): each wing's mean resultant is applied at its wing-root hinge
location (WingSpec.hinge_location), expressed in B. The fixed-body M1 run has
the body at identity pose, so world-frame record forces equal body-frame
forces.
"""

from __future__ import annotations

import json
from pathlib import Path

from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.experiment.record import load_record


def compute_average_forces(record_dir: str | Path) -> dict:
    """Per-wing cycle-averaged resultant force from a fixed-body run record."""
    ts, meta = load_record(record_dir)
    if ts.meta.get("kind") != "fixed_body":
        raise ValueError(f"record {record_dir} is not a fixed_body run")
    period = ts.meta["wingbeat_period_s"]
    wings = sorted({c.split("/")[0] for c in ts.channel_names()})
    per_wing = {}
    for w in wings:
        mean, spread = wingbeat_mean(ts, f"{w}/F_total", period)
        per_wing[w] = {"F_avg": list(map(float, mean)),
                       "cycle_spread": float(spread.max())}
    return {
        "source_record": str(record_dir),
        "vehicle": ts.meta.get("vehicle"),
        "flap_frequency_hz": ts.meta.get("flap_frequency_hz"),
        "per_wing": per_wing,
        "convention": "mean resultant applied at wing-root hinge, expressed in B",
        "assumptions": ["A-007"],
    }


def save_average_forces(record_dir: str | Path, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute_average_forces(record_dir), indent=2))
    return out


def load_average_forces(
    path: str | Path, vehicle_name: str, flap_frequency_hz: float | None = None
) -> tuple[list[tuple[str, list[float]]], list[str]]:
    """Load averaged forces; returns ([(wing_name, F_avg_B)], warnings)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    warnings = []
    if data.get("vehicle") != vehicle_name:
        warnings.append(
            f"averaged-force file vehicle {data.get('vehicle')!r} != current {vehicle_name!r}"
        )
    f = data.get("flap_frequency_hz")
    if flap_frequency_hz is not None and f is not None and f != flap_frequency_hz:
        warnings.append(
            f"averaged-force flap frequency {f} Hz != current {flap_frequency_hz} Hz"
        )
    return [(w, v["F_avg"]) for w, v in data["per_wing"].items()], warnings
