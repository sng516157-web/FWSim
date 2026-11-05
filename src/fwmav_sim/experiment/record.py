"""Experiment run records (M1 scope; extended in M3).

Every run saves (brief §8.2, action plan §3.3): the verbatim experiment and
vehicle YAML, metadata (git hash, python/package versions, timestamp,
assumption IDs, validity warnings) and the full time-resolved output as npz.
Headless rerun of the same YAML must reproduce the record (M3 tests this for
the ODE case; M1 tests repeat-run determinism).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import fwmav_sim
from fwmav_sim.core.timeseries import TimeSeries


def git_hash(cwd: str | Path = ".") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def save_record(
    out_root: str | Path,
    name: str,
    ts: TimeSeries,
    experiment_yaml: Path,
    vehicle_yaml: Path | None = None,
    extra_meta: dict | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = Path(out_root) / f"{stamp}_{name}"
    record_dir.mkdir(parents=True, exist_ok=False)

    record_dir.joinpath("config.yaml").write_text(
        Path(experiment_yaml).read_text(encoding="utf-8"), encoding="utf-8"
    )
    if vehicle_yaml is not None:
        record_dir.joinpath("vehicle.yaml").write_text(
            Path(vehicle_yaml).read_text(encoding="utf-8"), encoding="utf-8"
        )

    metadata = {
        "name": name,
        "timestamp": stamp,
        "git_hash": git_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "fwmav_sim_version": fwmav_sim.__version__,
        "time_series_meta": ts.meta,
        **(extra_meta or {}),
    }
    record_dir.joinpath("metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    np.savez_compressed(record_dir / "timeseries.npz", t=ts.t, **ts.channels)
    return record_dir


def load_record(record_dir: str | Path) -> tuple[TimeSeries, dict]:
    """Reload a run record: TimeSeries + metadata dict."""
    record_dir = Path(record_dir)
    npz = np.load(record_dir / "timeseries.npz")
    t = npz["t"]
    channels = {k: npz[k] for k in npz.files if k != "t"}
    metadata = json.loads(record_dir.joinpath("metadata.json").read_text(encoding="utf-8"))
    ts = TimeSeries(t=t, channels=channels, meta=metadata.get("time_series_meta", {}))
    return ts, metadata


class SessionRecorder:
    """Incremental session recorder (live sessions, M6.75+).

    A live session IS the simulation; its record is a full run record written
    on stop — identical schema to `fwmav-sim run` output. Samples accumulate
    in memory (per recorded sample: state + diagnostics); save() delegates to
    save_record.
    """

    def __init__(self, name: str, experiment_yaml: Path, vehicle_yaml: Path | None,
                 ts_meta: dict, extra_meta: dict | None = None):
        self._name = name
        self._exp = Path(experiment_yaml)
        self._veh = Path(vehicle_yaml) if vehicle_yaml else None
        self._ts_meta = ts_meta
        self._extra = extra_meta or {}
        self._t: list[float] = []
        self._channels: dict[str, list[np.ndarray]] = {}
        self._n = 0

    def add_sample(self, t: float, channels: dict[str, np.ndarray]) -> None:
        self._t.append(float(t))
        self._n += 1
        for k, v in channels.items():
            self._channels.setdefault(k, []).append(np.asarray(v, dtype=float))

    @property
    def n_samples(self) -> int:
        return self._n

    def save(self, out_root: str | Path = "outputs") -> Path | None:
        if not self._t:
            return None
        ts = TimeSeries(t=np.asarray(self._t),
                        channels={k: np.stack(v) for k, v in self._channels.items()},
                        meta=self._ts_meta)
        return save_record(out_root, self._name, ts, self._exp, self._veh, self._extra)
