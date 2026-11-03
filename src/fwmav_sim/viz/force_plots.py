"""Matplotlib force plots for fixed-body runs (M1 Task 9).

Every figure is generated from the stored run record (timeseries.npz) —
no hard-coded demo curves (brief §8.2, M1 stage file Task 9).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; figures are saved, never shown
import matplotlib.pyplot as plt
import numpy as np

from fwmav_sim.analysis.averaging import phase_average, wingbeat_mean
from fwmav_sim.core.timeseries import TimeSeries

_TERMS = ("F_trans", "F_rot", "F_am")
_TERM_LABELS = {"F_trans": "translational", "F_rot": "rotational", "F_am": "added mass"}


def _wings(ts: TimeSeries) -> list[str]:
    return sorted({name.split("/")[0] for name in ts.channel_names()})


def plot_forces_vs_phase(ts: TimeSeries, out_dir: Path) -> list[Path]:
    """Fx and Fz vs wingbeat phase with per-term overlay, per wing."""
    period = ts.meta["wingbeat_period_s"]
    paths = []
    for wing in _wings(ts):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for ax, comp, label in zip(axes, (0, 2), ("F_x", "F_z"), strict=True):
            for term in _TERMS:
                phase, prof, _ = phase_average(ts, f"{wing}/{term}", period)
                ax.plot(phase, prof[:, comp], label=_TERM_LABELS[term], linewidth=1)
            phase, prof, _ = phase_average(ts, f"{wing}/F_total", period)
            ax.plot(phase, prof[:, comp], "k", linewidth=2, label="resultant")
            ax.set_ylabel(f"{label} [N]")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"{wing}: forces vs wingbeat phase")
        axes[0].legend()
        axes[1].set_xlabel("wingbeat phase (0 = forward stroke extreme)")
        out = out_dir / f"{wing}_forces_vs_phase.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def plot_cycle_averaged_bars(ts: TimeSeries, out_dir: Path) -> Path:
    """Cycle-averaged force components (per term) and aerodynamic power, per wing."""
    period = ts.meta["wingbeat_period_s"]
    wings = _wings(ts)
    groups = ["F_x", "F_y", "F_z", "P_aero"]
    series = [*(f"{t} [{_TERM_LABELS[t]}]" for t in _TERMS), "F_total"]
    x = np.arange(len(groups))
    width = 0.8 / (len(series) * len(wings))
    fig, ax = plt.subplots(figsize=(9, 5))
    k = 0
    for wing in wings:
        for si, key in enumerate(series):
            vals = []
            for comp, label in ((0, "F_x"), (1, "F_y"), (2, "F_z")):
                term = key.split(" ")[0]
                mean, _ = wingbeat_mean(ts, f"{wing}/{term}", period)
                vals.append(float(mean[comp]))
            mean_p, _ = wingbeat_mean(ts, f"{wing}/P_aero", period)
            vals.append(float(mean_p))
            label = key if len(wings) == 1 else f"{key} — {wing}"
            ax.bar(x + (k - (len(series) * len(wings) - 1) / 2) * width, vals,
                   width=width, label=label)
            k += 1
    ax.set_xticks(x, groups)
    ax.set_ylabel("cycle-averaged value [N] / [W]")
    ax.set_title("Cycle-averaged forces and aerodynamic power")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    out = out_dir / "cycle_averaged_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spanwise_distribution(ts: TimeSeries, out_dir: Path, n_phases: int = 4) -> list[Path]:
    """Per-element spanwise force magnitude at selected wingbeat phases."""
    period = ts.meta["wingbeat_period_s"]
    paths = []
    for wing in _wings(ts):
        ef = ts[f"{wing}/element_force"]
        n_el = ef.shape[1]
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in range(n_phases):
            phase_frac = k / n_phases
            i = int(np.argmin(np.abs((ts.t % period) / period - phase_frac)))
            mag = np.linalg.norm(ef[i], axis=1)
            ax.plot(np.arange(1, n_el + 1), mag, "o-", label=f"phase {phase_frac:.2f}")
        ax.set_xlabel("blade element index (root -> tip)")
        ax.set_ylabel("|F_element| [N]")
        ax.set_title(f"{wing}: spanwise force distribution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = out_dir / f"{wing}_spanwise_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def make_report_figures(ts: TimeSeries, out_dir: str | Path) -> list[Path]:
    """Regenerate all report figures from a stored run record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_forces_vs_phase(ts, out_dir)
    paths.append(plot_cycle_averaged_bars(ts, out_dir))
    paths.extend(plot_spanwise_distribution(ts, out_dir))
    return paths
