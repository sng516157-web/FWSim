"""Runtime benchmarking (M6.75 Task 5; He's-FlapSim-style speed accounting).

The project has never measured its own speed. This command times the
dispatch of experiment YAMLs and reports wall time, simulated time,
sim/real-time ratio, and the active physics blocks — the speed side of the
fidelity/cost trade that M7 formalizes further.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fwmav_sim.experiment.config import load_experiment
from fwmav_sim.simulation.run import dispatch_experiment

DEFAULT_SUITE = [
    "experiments/m1_fixed_body_hover.yaml",
    "experiments/m3_free_body_hover.yaml",
    "experiments/m6_unsteady_airpulse.yaml",
    "experiments/m6_closed_loop_hover.yaml",
]


def _active_blocks(exp) -> list[str]:
    blocks = [exp.kind]
    vehicle_note = []
    if exp.isolation:
        iso = exp.isolation.model_dump()
        vehicle_note += [k for k, v in iso.items() if v and k != "wing_inertia_enabled"]
    if exp.kinematics is not None:
        blocks.append("prescribed_kinematics")
    if exp.closed_loop:
        blocks.append("closed_loop_lqr")
    return blocks + vehicle_note


def run_benchmark(experiments: list[str | Path] | None = None,
                  out_json: str | Path | None = None) -> list[dict]:
    """Time each experiment; returns rows and optionally writes JSON."""
    rows = []
    for path in experiments or DEFAULT_SUITE:
        path = Path(path)
        exp = load_experiment(path)
        t0 = time.perf_counter()
        ts, _, _ = dispatch_experiment(path)
        wall = time.perf_counter() - t0
        sim_t = float(ts.t[-1] - ts.t[0]) if len(ts.t) else 0.0
        rows.append({
            "experiment": exp.name,
            "kind": exp.kind,
            "sim_time_s": sim_t,
            "wall_time_s": wall,
            "sim_x_realtime": sim_t / wall if wall > 0 else float("inf"),
            "blocks": _active_blocks(exp),
        })
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'experiment':<28} {'kind':<18} {'sim [s]':>8} {'wall [s]':>9} "
             f"{'x realtime':>10}  blocks",
             "-" * 100]
    for r in rows:
        lines.append(f"{r['experiment']:<28} {r['kind']:<18} {r['sim_time_s']:>8.3f} "
                     f"{r['wall_time_s']:>9.2f} {r['sim_x_realtime']:>10.3f}  "
                     f"{', '.join(r['blocks'])}")
    return "\n".join(lines)
