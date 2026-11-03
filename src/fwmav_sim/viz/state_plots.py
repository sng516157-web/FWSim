"""State-trace plots for dynamic (M2) run records: pendulum and free reduced.

All figures derive from the stored run record (brief §8.2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fwmav_sim.core.timeseries import TimeSeries


def make_state_figures(ts: TimeSeries, out_dir: str | Path) -> list[Path]:
    """State vs time, abdomen command tracking, and energy drift."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = ts.meta.get("kind", "")
    paths: list[Path] = []
    t = ts.t

    if kind == "suspended_pendulum":
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t, np.degrees(ts["phi_p"]))
        axes[0].set_ylabel("swing φ_p [deg]")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, np.degrees(ts["alpha_abd"]), label="actual")
        if "alpha_commanded" in ts.channel_names():
            axes[1].plot(t, np.degrees(ts["alpha_commanded"]), "--", label="commanded")
        axes[1].set_ylabel("abdomen α_abd [deg]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        e = ts["energy_total"]
        axes[2].plot(t, e - e[0])
        axes[2].set_ylabel("ΔE [J]")
        axes[2].set_xlabel("t [s]")
        axes[2].grid(True, alpha=0.3)
        out = out_dir / "pendulum_states.png"
    elif kind == "free_reduced":
        fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(t, ts["x"], label="x")
        axes[0].plot(t, ts["z"], label="z (down)")
        axes[0].set_ylabel("position [m]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, np.degrees(ts["theta"]))
        axes[1].set_ylabel("pitch θ [deg]")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(t, np.degrees(ts["alpha_abd"]))
        axes[2].set_ylabel("abdomen α_abd [deg]")
        axes[2].grid(True, alpha=0.3)
        e = ts["energy_total"]
        axes[3].plot(t, e - e[0])
        axes[3].set_ylabel("ΔE [J]")
        axes[3].set_xlabel("t [s]")
        axes[3].grid(True, alpha=0.3)
        out = out_dir / "free_reduced_states.png"
    else:
        return paths

    fig.suptitle(f"{ts.meta.get('name', kind)} — {kind}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)
    return paths
