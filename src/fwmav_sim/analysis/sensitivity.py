"""One-at-a-time parameter sensitivity analysis (post-M5 item 2).

Question answered: which physical parameters actually move the outputs — i.e.
what is worth measuring carefully on a real vehicle. Each selected scalar
parameter is perturbed by +/- rel_step (central difference), the experiment is
re-run in process, and the normalized sensitivity

    S_i = (d metric / metric_0) / (d p_i / p_i_0)

is reported per metric. Metrics depend on the experiment kind:

  fixed_body : total cycle-mean force vector (both wings summed), cycle-mean
               aerodynamic power, per-wing cycle-mean vertical force
  dynamic    : final position norm and peak |z| excursion (free flight /
               pendulum); FAILED runs record NaN with the failure status

Parameter addressing (dot paths into the experiment/vehicle YAML trees):
  vehicle:bodies.<name>.mass                     scalar body parameter
  vehicle:wings.<name>.<field>                   e.g. span, root_chord, mass
  kinematics:<field>                             broadcast kinematics
  kinematics:<wing>.<field>                      per-wing kinematics
  environment:<field>                            air_density, ...

Only scalar ``Parameter`` leaves (with a ``value``) are addressable. The
perturbation is applied to the raw YAML mapping and re-validated through the
schema, so provenance structure is preserved and invalid perturbations fail
loudly at load.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.core.timeseries import TimeSeries


@dataclass(frozen=True)
class SensitivityEntry:
    """One parameter's effect on the metrics."""

    path: str
    base_value: float
    step: float  # absolute perturbation used
    derivatives: dict[str, float | list[float]]  # unnormalized d metric / d p
    sensitivities: dict[str, float | list[float]]  # normalized S per metric
    status: str  # "OK" / "OK:one-sided" / "FAILED:<message>"


def _resolve(raw: dict, path: str) -> dict:
    """Return the Parameter mapping at a dot path (''-separated segments
    support list indices by name key, e.g. bodies.body)."""
    node = raw
    for seg in path.split("."):
        if isinstance(node, list):
            node = next(item for item in node
                        if isinstance(item, dict) and item.get("name") == seg)
        else:
            node = node[seg]
    return node


def _get_value(exp_raw: dict, veh_raw: dict, path: str) -> float:
    root, _, rest = path.partition(":")
    raw = {"vehicle": veh_raw, "kinematics": exp_raw.get("kinematics"),
           "environment": exp_raw.get("environment")}[root]
    return float(_resolve(raw, rest)["value"])


def _set_value(exp_raw: dict, veh_raw: dict, path: str, value: float) -> None:
    root, _, rest = path.partition(":")
    raw = {"vehicle": veh_raw, "kinematics": exp_raw.get("kinematics"),
           "environment": exp_raw.get("environment")}[root]
    _resolve(raw, rest)["value"] = value


def default_parameters(exp_raw: dict, veh_raw: dict) -> list[str]:
    """Every scalar Parameter of the vehicle bodies/wings, the environment,
    and the kinematics — the 'auto' parameter set."""
    paths: list[str] = []
    for group in ("bodies", "wings"):
        for item in veh_raw.get(group, []):
            for key, val in item.items():
                if isinstance(val, dict) and "value" in val \
                        and np.isscalar(val["value"]):
                    if item.get("massless") and val["value"] == 0:
                        continue  # massless frame carriers: nothing to perturb
                    paths.append(f"vehicle:{group}.{item['name']}.{key}")
    for key, val in exp_raw.get("environment", {}).items():
        if isinstance(val, dict) and np.isscalar(val.get("value")):
            paths.append(f"environment:{key}")
    kin = exp_raw.get("kinematics") or {}
    if isinstance(kin, dict) and "flap_amplitude" in kin:  # broadcast
        for key, val in kin.items():
            if isinstance(val, dict) and np.isscalar(val.get("value")):
                paths.append(f"kinematics:{key}")
    elif isinstance(kin, dict):  # per-wing mapping
        for wname, kmap in kin.items():
            for key, val in (kmap or {}).items():
                if isinstance(val, dict) and np.isscalar(val.get("value")):
                    paths.append(f"kinematics:{wname}.{key}")
    return paths


def _metrics(ts: TimeSeries, kind: str, meta: dict) -> dict[str, float | list[float]]:
    """Output metrics per experiment kind (module docstring)."""
    if kind == "fixed_body":
        period = ts.meta["wingbeat_period_s"]
        wings = sorted({c.split("/")[0] for c in ts.channel_names() if "/" in c})
        f_tot = np.zeros(3)
        out: dict[str, float | list[float]] = {}
        for w in wings:
            mean, _ = wingbeat_mean(ts, f"{w}/F_total", period)
            f_tot += np.asarray(mean)
            out[f"{w}.meanFz"] = float(np.asarray(mean)[2])
            p_mean, _ = wingbeat_mean(ts, f"{w}/P_aero", period)
            out[f"{w}.meanP"] = float(p_mean)
        out["mean_force"] = [float(x) for x in f_tot]
        out["meanFz_total"] = float(f_tot[2])
        return out
    # dynamic kinds: final position norm + peak |z|
    if "p" in ts.channel_names():
        p = ts["p"]
        return {"final_|p|": float(np.linalg.norm(p[-1])),
                "peak_|z|": float(np.abs(p[:, 2]).max())}
    if "z" in ts.channel_names():
        z = ts["z"]
        return {"final_|z|": float(abs(z[-1])), "peak_|z|": float(np.abs(z).max())}
    raise ValueError(f"no metrics for kind {kind!r}")


def _run_patched(exp_raw: dict, veh_raw: dict, tmp_dir: Path) -> tuple[TimeSeries, str]:
    """Write patched YAMLs to tmp_dir and run in process (no record saved)."""
    from fwmav_sim.simulation.run import dispatch_experiment

    veh_path = tmp_dir / "vehicle.yaml"
    exp_path = tmp_dir / "experiment.yaml"
    veh_path.write_text(yaml.safe_dump(veh_raw), encoding="utf-8")
    exp_raw = {**exp_raw, "vehicle": "vehicle.yaml"}
    exp_path.write_text(yaml.safe_dump(exp_raw), encoding="utf-8")
    ts, _, meta = dispatch_experiment(exp_path)
    return ts, meta.get("status", "OK")


def sensitivity_analysis(
    experiment_path: str | Path,
    params: list[str] | None = None,
    rel_step: float = 0.05,
    tmp_dir: str | Path | None = None,
) -> tuple[dict[str, float | list[float]], list[SensitivityEntry]]:
    """Central-difference OAT sensitivities of the experiment's metrics.

    Returns (baseline_metrics, entries). Uses a temporary directory for the
    patched YAMLs (system temp if tmp_dir is None).
    """
    import tempfile

    experiment_path = Path(experiment_path)
    exp_raw = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    veh_raw = yaml.safe_load(
        (experiment_path.parent / exp_raw["vehicle"]).read_text(encoding="utf-8"))
    if params is None:
        params = default_parameters(exp_raw, veh_raw)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        tmp = Path(td)
        ts0, status0 = _run_patched(exp_raw, veh_raw, tmp)
        base = _metrics(ts0, exp_raw["kind"], ts0.meta)

        entries: list[SensitivityEntry] = []
        for i, path in enumerate(params):
            p0 = _get_value(exp_raw, veh_raw, path)
            dp = rel_step * (abs(p0) if p0 != 0.0 else 1.0)

            def _eval(value: float, tag: str):
                sub = tmp / f"{tag}{i}"
                sub.mkdir()
                patched = copy.deepcopy((exp_raw, veh_raw))
                _set_value(patched[0], patched[1], path, value)
                ts, st = _run_patched(patched[0], patched[1], sub)
                if st != "OK":
                    raise RuntimeError(st)
                return _metrics(ts, exp_raw["kind"], ts.meta)

            # central difference; one-sided fallback when a perturbation is
            # schema-invalid (e.g. p0 - dp < 0 for a non-negative parameter)
            try:
                m_p = _eval(p0 + dp, "p")
            except Exception:
                m_p = None
            try:
                m_m = _eval(p0 - dp, "m")
            except Exception:
                m_m = None
            if m_p is None and m_m is None:
                entries.append(SensitivityEntry(
                    path=path, base_value=p0, step=dp, derivatives={}, sensitivities={},
                    status="FAILED:both perturbations invalid"))
                continue
            status = "OK" if (m_p is not None and m_m is not None) else "OK:one-sided"
            derivs: dict[str, float | list[float]] = {}
            sens: dict[str, float | list[float]] = {}
            for key in base:
                b = np.asarray(base[key], dtype=float)
                if m_p is not None and m_m is not None:
                    d = (np.asarray(m_p[key]) - np.asarray(m_m[key])) / (2.0 * dp)
                elif m_p is not None:
                    d = (np.asarray(m_p[key]) - b) / dp
                else:
                    d = (b - np.asarray(m_m[key])) / dp
                derivs[key] = float(d) if d.ndim == 0 else [float(x) for x in d]
                if p0 != 0.0:
                    rel = d * p0 / np.where(np.abs(b) > 1e-15, np.abs(b), np.nan)
                    sens[key] = float(rel) if rel.ndim == 0 else [float(x) for x in rel]
                else:
                    # normalized sensitivity undefined at a zero base value
                    sens[key] = float("nan") if d.ndim == 0 else [float("nan")] * len(d)
            entries.append(SensitivityEntry(path=path, base_value=p0, step=dp,
                                            derivatives=derivs, sensitivities=sens,
                                            status=status))
    return base, entries


def report_markdown(experiment: str, base: dict, entries: list[SensitivityEntry],
                    metric: str = "meanFz_total") -> str:
    """Ranked Markdown table for one scalar metric (default: total mean Fz)."""
    rows = []
    for e in entries:
        s = e.sensitivities.get(metric, float("nan"))
        arr = np.asarray(s, dtype=float)
        if np.all(np.isfinite(arr)):
            shown = f"{float(np.linalg.norm(arr)):.3g}" if arr.ndim else f"{float(arr):.3g}"
            s_abs = float(np.linalg.norm(arr))
        else:
            # normalized S undefined (zero base value) — rank by the
            # unnormalized derivative instead
            d = np.asarray(e.derivatives.get(metric, float("nan")), dtype=float)
            shown = f"d/dp = {float(np.linalg.norm(d)):.3g}"
            s_abs = float(np.linalg.norm(d)) if np.all(np.isfinite(d)) else float("nan")
        rows.append((e.path, e.base_value, shown, s_abs, e.status))
    rows.sort(key=lambda r: -abs(r[3]) if np.isfinite(r[3]) else 0.0)
    lines = [f"# Sensitivity — {experiment} (metric: {metric})", "",
             "| rank | parameter | base value | S (normalized; d/dp if undefined) | status |",
             "|---|---|---|---|---|"]
    for i, (path, p0, s_str, _, status) in enumerate(rows, 1):
        lines.append(f"| {i} | `{path}` | {p0:.6g} | {s_str} | {status} |")
    return "\n".join(lines) + "\n"
