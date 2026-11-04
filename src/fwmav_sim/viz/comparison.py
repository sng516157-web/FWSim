"""Comparison figures (M4 Task 5): overlay on the phase grid, residual time
series, Welch spectra. All from aligned series passed in by the compare
pipeline — nothing hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_comparison_figures(
    phase: np.ndarray,
    sim: np.ndarray,
    exp: np.ndarray,
    t: np.ndarray,
    sim_full: np.ndarray,
    exp_aligned: np.ndarray,
    freqs: np.ndarray,
    psd_sim: np.ndarray,
    psd_exp: np.ndarray,
    channel: str,
    out_dir: str | Path,
    labels: tuple[str, str] = ("simulated", "experimental"),
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = channel.replace("/", "_")
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phase, sim, label=labels[0])
    ax.plot(phase, exp, "--", label=labels[1])
    ax.set_xlabel("wingbeat phase")
    ax.set_ylabel(channel)
    ax.set_title(f"{channel}: phase-averaged overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = out_dir / f"overlay_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, sim_full, label=labels[0])
    ax.plot(t, exp_aligned, "--", label=f"{labels[1]} (aligned)")
    ax.plot(t, sim_full - exp_aligned, "r", linewidth=0.8, label="residual")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(channel)
    ax.set_title(f"{channel}: time series and residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = out_dir / f"residual_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freqs, psd_sim, label=labels[0])
    ax.semilogy(freqs, psd_exp, "--", label=labels[1])
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"{channel}: Welch spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = out_dir / f"spectra_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)
    return paths
