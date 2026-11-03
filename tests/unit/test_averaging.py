"""§11.1 wingbeat averaging tests (M1 Task 8)."""

from __future__ import annotations

import numpy as np

from fwmav_sim.analysis.averaging import cycle_bounds, phase_average, wingbeat_mean
from fwmav_sim.core.timeseries import TimeSeries


def _ts(t, data) -> TimeSeries:
    return TimeSeries(t=np.asarray(t, float), channels={"x": np.asarray(data, dtype=float)})


def test_wingbeat_average_constant_signal():
    t = np.linspace(0, 1.0, 1001)
    ts = _ts(t, np.full_like(t, 3.7))
    mean, spread = wingbeat_mean(ts, "x", period=0.1)
    assert abs(mean - 3.7) < 1e-15
    assert spread < 1e-12  # float round-off in cycle-mean reduction


def test_phase_segmentation():
    """Sinusoid: cycle mean ~ 0; phase-averaged profile recovers the signal."""
    period = 0.05
    n_per_cycle = 400
    t = np.arange(0, 20 * period, period / n_per_cycle)
    signal = np.sin(2 * np.pi * t / period)
    ts = _ts(t, signal)

    mean, spread = wingbeat_mean(ts, "x", period)
    assert abs(mean) < 1e-3
    assert spread < 1e-3

    # one bin per sample so spread measures cycle-to-cycle variation only
    phase, prof, prof_spread = phase_average(ts, "x", period, n_phase_bins=n_per_cycle)
    expected = np.sin(2 * np.pi * phase)
    err = np.nanmean(np.abs(prof - expected))
    assert err < 0.02  # binning accuracy
    assert np.nanmedian(prof_spread) < 1e-8  # periodic signal: identical cycles


def test_cycle_bounds_complete_cycles_only():
    t = np.arange(0, 0.23, 0.001)  # 2.3 cycles of period 0.1
    bounds = cycle_bounds(t, 0.1)
    assert len(bounds) == 2  # trailing partial cycle excluded


def test_wingbeat_mean_per_cycle_spread_detects_transient():
    """A growing signal must show nonzero per-cycle spread (brief §5.5)."""
    period = 0.1
    t = np.arange(0, 5 * period, period / 100)
    ts = _ts(t, np.exp(t))
    _, spread = wingbeat_mean(ts, "x", period)
    assert spread > 0.05
