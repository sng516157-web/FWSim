"""Wingbeat-phase synchronization (M4 Task 2).

Estimate the dominant flapping harmonic by FFT, cross-correlate the simulated
vs experimental reference channel to find the time offset, report offset and
residual drift — as outputs, never as silent corrections (brief §5.9).
"""

from __future__ import annotations

import numpy as np


def dominant_frequency(t: np.ndarray, x: np.ndarray) -> float:
    """Dominant non-DC frequency via FFT on the de-meaned, Hann-windowed signal."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    dt = float(np.median(np.diff(t)))
    w = np.hanning(len(x))
    X = np.fft.rfft((x - x.mean()) * w)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    if len(freqs) < 2:
        raise ValueError("signal too short for frequency estimate")
    return float(freqs[1:][np.argmax(np.abs(X[1:]))])


def estimate_offset(
    t_sim: np.ndarray, x_sim: np.ndarray,
    t_exp: np.ndarray, x_exp: np.ndarray,
    reference_frequency: float,
) -> tuple[float, float]:
    """Wingbeat-phase offset of exp relative to sim via cross-correlation.

    Periodic signals are ambiguous modulo the wingbeat period T; the returned
    offset is the physically meaningful phase offset wrapped to (-T/2, T/2],
    refined to sub-sample precision by parabolic interpolation. The integer-
    cycle ambiguity is inherent — comparisons must use phase, not absolute
    time, and the report must state this. Returns (phase_offset_seconds,
    drift_seconds_per_s).
    """
    t_sim = np.asarray(t_sim, float)
    x_sim = np.asarray(x_sim, float)
    t_exp = np.asarray(t_exp, float)
    x_exp = np.asarray(x_exp, float)

    dt = 1.0 / (20.0 * reference_frequency)
    t0 = max(t_sim[0], t_exp[0])
    t1 = min(t_sim[-1], t_exp[-1])
    if t1 <= t0:
        raise ValueError("no time overlap between simulated and experimental series")
    grid = np.arange(t0, t1, dt)
    s = np.interp(grid, t_sim, x_sim)
    e = np.interp(grid, t_exp, x_exp)
    s = s - s.mean()
    e = e - e.mean()

    corr = np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(e))).real
    corr = np.concatenate([corr[len(corr) // 2:], corr[: len(corr) // 2]])
    lags = np.arange(-len(grid) // 2, len(grid) // 2) * dt
    i = int(np.argmax(corr))
    # parabolic sub-sample refinement
    if 0 < i < len(corr) - 1:
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-300 else 0.0
    else:
        shift = 0.0
    raw_offset = lags[i] + shift * dt
    # Periodic signals are ambiguous modulo the wingbeat period; the physically
    # meaningful quantity is the phase offset. We report the offset wrapped to
    # (-T/2, T/2] and flag the integer-cycle ambiguity in the report.
    period = 1.0 / reference_frequency
    offset = float(((raw_offset + period / 2) % period) - period / 2)

    # drift: offset of the second half minus offset of the first half
    half = len(grid) // 2
    drift = 0.0
    if half > 100:
        o1 = _offset_of(grid[:half], s[:half], e[:half], dt)
        o2 = _offset_of(grid[half:], s[half:], e[half:], dt)
        span = grid[half:][0] - grid[:half][0]
        drift = float((o2 - o1) / span) if span > 0 else 0.0
    return offset, drift


def _offset_of(t, s, e, dt):
    s = s - s.mean()
    e = e - e.mean()
    corr = np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(e))).real
    corr = np.concatenate([corr[len(corr) // 2:], corr[: len(corr) // 2]])
    lags = np.arange(-len(t) // 2, len(t) // 2) * dt
    return float(lags[int(np.argmax(corr))])


def phase_grid(t: np.ndarray, period: float, n_bins: int = 100) -> np.ndarray:
    """Normalized wingbeat phase in [0, 1) for each sample."""
    return ((np.asarray(t, float) - t[0]) / period) % 1.0


def resample_to_phase(
    t: np.ndarray, x: np.ndarray, period: float, n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-binned mean of x (normalized phase grid, bin centres b/N)."""
    phase = phase_grid(t, period)
    idx = np.floor((phase + 0.5 / n_bins) * n_bins).astype(int) % n_bins
    out = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        sel = x[idx == b]
        if len(sel):
            out[b] = sel.mean()
            cnt[b] = len(sel)
    centres = np.arange(n_bins) / n_bins
    return centres, out
