"""Ablation report generator (M6.75 Task 6; He's-FlapSim-style ablation).

Every optional physics block must earn its place with a measured on/off
comparison (the S6 lesson). Given an experiment, this runs it once per
toggleable block — unsteady aero terms, wing structural inertia — with the
block removed, and tabulates the change in the run's headline metrics
(cycle-mean forces for fixed_body; final z/|p| for dynamic kinds).
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import yaml

from fwmav_sim.analysis.averaging import wingbeat_mean


def _blocks(exp_raw: dict, veh_raw: dict) -> list[tuple[str, callable]]:
    """(label, patch_fn) per toggleable block present in this experiment."""
    out = []
    if (veh_raw.get("aero") or {}).get("unsteady"):
        def drop_unsteady(e, v):
            v["aero"]["unsteady"] = {}
        out.append(("unsteady_terms_off", drop_unsteady))
    if any("mass" in w for w in veh_raw.get("wings", [])):
        def drop_wing_mass(e, v):
            for w in v["wings"]:
                w.pop("mass", None)
        out.append(("wing_inertia_off", drop_wing_mass))
    if veh_raw.get("abdomen_aero") is not None:
        def drop_abdomen_aero(e, v):
            v["abdomen_aero"] = None
            iso = e.setdefault("isolation", {})
            iso["abdomen_aero_enabled"] = False
        out.append(("abdomen_aero_off", drop_abdomen_aero))
    return out


def _metrics(ts) -> dict[str, float]:
    kind = ts.meta.get("kind")
    if kind == "fixed_body":
        period = ts.meta["wingbeat_period_s"]
        wings = sorted({c.split("/")[0] for c in ts.channel_names() if "/" in c})
        out = {}
        for w in wings:
            mean, _ = wingbeat_mean(ts, f"{w}/F_total", period)
            out[f"{w}.meanFx"] = float(mean[0])
            out[f"{w}.meanFz"] = float(mean[2])
        return out
    if "p" in ts.channel_names():
        return {"final_|p|": float(np.linalg.norm(ts["p"][-1])),
                "peak_z": float(ts["p"][:, 2].max())}
    return {}


def run_ablation(experiment_path: str | Path) -> tuple[dict, list[dict]]:
    """(baseline_metrics, rows) — one row per toggled block."""
    experiment_path = Path(experiment_path)
    exp_raw = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    veh_raw = yaml.safe_load(
        (experiment_path.parent / exp_raw["vehicle"]).read_text(encoding="utf-8"))
    blocks = _blocks(exp_raw, veh_raw)
    from fwmav_sim.simulation.run import dispatch_experiment

    def run_pair(e_raw, v_raw, tmp):
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "vehicle.yaml").write_text(yaml.safe_dump(v_raw), encoding="utf-8")
        (tmp / "experiment.yaml").write_text(
            yaml.safe_dump({**e_raw, "vehicle": "vehicle.yaml"}), encoding="utf-8")
        ts, _, _ = dispatch_experiment(tmp / "experiment.yaml")
        return _metrics(ts)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = run_pair(exp_raw, veh_raw, tmp / "base")
        rows = []
        for label, patch in blocks:
            e2, v2 = copy.deepcopy(exp_raw), copy.deepcopy(veh_raw)
            patch(e2, v2)
            sub = tmp / label
            sub.mkdir()
            m = run_pair(e2, v2, sub)
            rows.append({"block": label,
                         "metrics": {k: float(m.get(k, np.nan)) for k in base},
                         "delta": {k: float(m.get(k, np.nan) - base[k]) for k in base}})
    return base, rows


def format_ablation(experiment: str, base: dict, rows: list[dict]) -> str:
    keys = list(base)
    lines = [f"# Ablation — {experiment}", "",
             "| variant | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    base_line = ["baseline (all blocks on)"] + [f"{base[k]:.4g}" for k in keys]
    lines.append("| " + " | ".join(base_line) + " |")
    for r in rows:
        lines.append("| " + r["block"] + " | "
                     + " | ".join(f"{r['metrics'][k]:+.4g} ({r['delta'][k]:+.3g})"
                                 for k in keys) + " |")
    lines.append("")
    lines.append("(value (delta vs baseline))")
    return "\n".join(lines) + "\n"
