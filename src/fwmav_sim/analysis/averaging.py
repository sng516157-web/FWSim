"""Wingbeat averaging (M1 Task 8).

Cycle averages and phase-resolved averages are derived from the same
time-resolved arrays the simulation produced (brief §3.3). Phase is
phi_phase = (t mod T) / T with T the wingbeat period; per-cycle spread is
reported alongside the mean so non-periodic transients are visible
(brief §5.5: do not assume periodic steady state).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.timeseries import TimeSeries


def cycle_bounds(t: np.ndarray, period: float) -> list[tuple[int, int]]:
    """Index ranges [start, end) of each complete wingbeat cycle in t."""
    t0 = t[0]
    n_cycles = int((t[-1] - t0) / period)
    bounds = []
    for k in range(n_cycles):
        mask = np.nonzero((t >= t0 + k * period) & (t < t0 + (k + 1) * period))[0]
        if len(mask) > 0:
            bounds.append((int(mask[0]), int(mask[-1]) + 1))
    return bounds


def wingbeat_mean(ts: TimeSeries, channel: str, period: float) -> tuple[float | np.ndarray, float]:
    """Cycle average of a channel plus the spread (std) of per-cycle means.

    Returns (mean, spread): mean over complete cycles of the per-cycle mean;
    spread = std of the per-cycle means (0 for a single cycle).
    """
    data = ts[channel]
    bounds = cycle_bounds(ts.t, period)
    if not bounds:
        raise ValueError("no complete wingbeat cycle in the time series")
    means = [data[a:b].mean(axis=0) for a, b in bounds]
    mean = np.mean(means, axis=0)
    spread = np.std(means, axis=0) if len(means) > 1 else np.zeros_like(mean)
    return mean, spread


def phase_average(
    ts: TimeSeries, channel: str, period: float, n_phase_bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-aligned average across cycles.

    Returns (phase, mean, spread) with phase in [0, 1) bin centres; mean and
    spread are per-bin statistics across cycles (spread = 0 where a bin sees
    one sample).
    """
    data = ts[channel]
    phase = ((ts.t - ts.t[0]) / period) % 1.0
    # bin by nearest centre: bin b covers [b/N - 1/(2N), b/N + 1/(2N)); robust to
    # samples landing exactly on bin edges (e.g. integer samples per cycle)
    centres = np.arange(n_phase_bins) / n_phase_bins
    idx = np.floor((phase + 0.5 / n_phase_bins) * n_phase_bins).astype(int) % n_phase_bins
    shape = (n_phase_bins,) + data.shape[1:]
    mean = np.full(shape, np.nan)
    spread = np.zeros(shape)
    for b in range(n_phase_bins):
        sel = data[idx == b]
        if len(sel) > 0:
            mean[b] = sel.mean(axis=0)
            spread[b] = sel.std(axis=0) if len(sel) > 1 else 0.0
    return centres, mean, spread
