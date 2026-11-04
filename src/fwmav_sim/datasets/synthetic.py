"""Synthetic sanity dataset generator (M4 Task 6).

Builds a known-truth "experiment" from a run record: downsamples a channel,
applies a known phase offset and known Gaussian noise, writes CSV + manifest
with label=calibration, synthetic=true. The manifest records the exact
offset/noise so the pipeline test can verify recovery. Never presented as
validation (brief §14.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fwmav_sim.experiment.record import load_record


def make_synthetic_dataset(
    record_dir: str | Path,
    out_dir: str | Path,
    channel: str,
    component: int | None = None,
    phase_offset_s: float = 0.012,
    noise_std: float = 0.02,
    downsample_hz: float = 222.0,
    seed: int = 42,
) -> Path:
    """Create a synthetic dataset directory from a run record channel.

    Returns the dataset directory. The known truth (offset, noise, source
    record) is written into the manifest's notes/provenance fields.
    """
    ts, meta = load_record(record_dir)
    x = ts[channel]
    if x.ndim > 1:
        if component is None:
            raise ValueError("vector channel: pass component=")
        x = x[:, component]
    t = ts.t

    dt = 1.0 / downsample_hz
    # anti-alias lowpass before downsampling (real ADCs band-limit; otherwise
    # harmonics above the new Nyquist fold into the fundamental and corrupt
    # phase-offset recovery)
    from scipy.signal import butter, filtfilt

    fs_full = 1.0 / float(np.median(np.diff(t)))
    b, a = butter(4, 0.45 * downsample_hz / (fs_full / 2))
    x_aa = filtfilt(b, a, x)
    # crop the central 80%: filtfilt's edge transient and any run startup
    # transient are excluded, as a rig would discard non-steady edges
    t_ds_full = np.arange(t[0], t[-1], dt)
    span = t[-1] - t[0]
    keep = (t_ds_full > t[0] + 0.1 * span) & (t_ds_full < t[-1] - 0.1 * span)
    t_ds = t_ds_full[keep]
    x_ds = np.interp(t_ds, t, x_aa)
    # apply known phase offset (shift content later by phase_offset_s)
    x_off = np.interp(t_ds + phase_offset_s, t, x_aa)
    rng = np.random.default_rng(seed)
    x_noisy = x_off + rng.normal(0.0, noise_std * float(np.abs(x_ds).max() or 1.0), len(t_ds))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": t_ds, channel.replace("/", "_"): x_noisy}).to_csv(
        out_dir / "data.csv", index=False)

    manifest = {
        "name": f"synthetic_{channel.replace('/', '_')}",
        "label": "calibration",
        "synthetic": True,
        "source": f"generated from run record {record_dir}",
        "provenance": "synthetic known-truth: NOT validation data (brief 14.5)",
        "sampling_rate_hz": downsample_hz,
        "columns": {
            "t": {"quantity": "time", "unit": "s", "frame": "W"},
            channel.replace("/", "_"): {
                "quantity": channel, "unit": "N", "frame": "W",
                "note": "simulated truth with known offset/noise",
            },
        },
        "notes": yaml.safe_dump({
            "known_phase_offset_s": phase_offset_s,
            "known_noise_std_rel": noise_std,
            "seed": seed,
            "source_record": str(record_dir),
        }),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return out_dir
