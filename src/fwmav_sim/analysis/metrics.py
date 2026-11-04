"""Comparison metrics (M4 Task 4).

Each function is small, hand-checkable and unit-tested against hand-computed
values. All take aligned, same-grid arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE normalized by the range (max - min) of the reference series b."""
    b = np.asarray(b, float)
    span = float(b.max() - b.min())
    if span == 0.0:
        raise ValueError("reference series has zero range; NRMSE undefined")
    return rmse(a, b) / span


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0.0 or b.std() == 0.0:
        raise ValueError("correlation undefined for a constant series")
    return float(np.corrcoef(a, b)[0, 1])


def phase_lag(t: np.ndarray, a: np.ndarray, b: np.ndarray, frequency: float) -> float:
    """Phase lag of b behind a at the given frequency, in radians, via the
    cross-spectrum phase (dominant-harmonic lag)."""
    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = np.hanning(n)
    A = np.fft.rfft((np.asarray(a, float) - np.mean(a)) * w)
    B = np.fft.rfft((np.asarray(b, float) - np.mean(b)) * w)
    freqs = np.fft.rfftfreq(n, d=dt)
    i = int(np.argmin(np.abs(freqs - frequency)))
    return float(np.angle(A[i] * np.conj(B[i])))


def welch_spectra(
    t: np.ndarray, a: np.ndarray, b: np.ndarray, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch PSDs of both series on a common frequency grid."""
    dt = float(np.median(np.diff(np.asarray(t, float))))
    nperseg = min(nperseg, len(t))
    f, pa = welch(np.asarray(a, float), fs=1.0 / dt, nperseg=nperseg)
    _, pb = welch(np.asarray(b, float), fs=1.0 / dt, nperseg=nperseg)
    return f, pa, pb
