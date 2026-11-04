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

    # Robust phase estimate: cross-spectrum phase at the dominant harmonic
    # (time-domain cross-correlation of harmonic-rich wingbeat signals picks
    # spurious peaks). Sign convention: e(t) = s(t + tau)  =>  tau as below.
    # Phase offset via least-squares projection onto a sin/cos template at the
    # reference frequency PLUS its first overtone (wingbeat force signals are
    # harmonic-rich; including 2*f0 removes leakage bias in non-integer
    # windows). This is the optimal phase estimator when the frequency is
    # known (from the sim record's metadata).
    def fundamental_phase_difference(a, b):
        n = len(a)
        tt = np.arange(n) * dt
        w = 2.0 * np.pi * reference_frequency
        A = np.column_stack([np.sin(w * tt), np.cos(w * tt),
                             np.sin(2 * w * tt), np.cos(2 * w * tt),
                             np.ones(n)])

        def phase(v):
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            return np.arctan2(coef[1], coef[0])  # phase of the fundamental

        return float(phase(b) - phase(a))

    def offset_of(a, b):
        dphi = fundamental_phase_difference(a, b)
        tau = dphi / (2 * np.pi * reference_frequency)
        # wrap to (-T/2, T/2]
        period = 1.0 / reference_frequency
        return float(((tau + period / 2) % period) - period / 2)

    offset = offset_of(s, e)
    half = len(grid) // 2
    drift = 0.0
    if half > 100:
        o1 = offset_of(s[:half], e[:half])
        o2 = offset_of(s[half:], e[half:])
        span = grid[half:][0] - grid[:half][0]
        drift = float((o2 - o1) / span) if span > 0 else 0.0
    return offset, drift




def phase_grid(t: np.ndarray, period: float, n_bins: int = 100) -> np.ndarray:
    """Normalized wingbeat phase in [0, 1) for each sample."""
    return ((np.asarray(t, float) - t[0]) / period) % 1.0


def resample_to_phase(
    t: np.ndarray, x: np.ndarray, period: float, n_bins: int = 100,
    t_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-binned mean of x (normalized phase grid, bin centres b/N).

    Phase is anchored at t_ref (default: the series' own first sample).
    Comparisons between series must pass a SHARED t_ref (e.g. the sim clock
    zero), otherwise each series anchors its phase at its own start and the
    profiles are rotated by the series-start difference.

    Empty bins (undersampled phase) are NaN — callers must mask NaNs, never
    treat them as zero.
    """
    anchor = t[0] if t_ref is None else t_ref
    phase = ((np.asarray(t, float) - anchor) / period) % 1.0
    idx = np.floor((phase + 0.5 / n_bins) * n_bins).astype(int) % n_bins
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = x[idx == b]
        if len(sel):
            out[b] = sel.mean()
    centres = np.arange(n_bins) / n_bins
    return centres, out
