"""§11.1 M4 dataset-machinery unit tests: importer, alignment, filtering, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from fwmav_sim.analysis.metrics import mae, nrmse, pearson, phase_lag, rmse, welch_spectra
from fwmav_sim.datasets.alignment import dominant_frequency, estimate_offset, resample_to_phase
from fwmav_sim.datasets.filtering import FilterSpec, apply_filter
from fwmav_sim.datasets.importer import load_dataset


def _write_dataset(tmp_path, label="calibration", unit="N", extra_col=False):
    d = tmp_path / "ds"
    d.mkdir()
    t = np.linspace(0, 1.0, 101)
    fx = np.sin(2 * np.pi * 5 * t)
    cols = {"t": t, "force_x": fx}
    if extra_col:
        cols["mystery"] = fx
    pd.DataFrame(cols).to_csv(d / "data.csv", index=False)
    manifest = {
        "name": "test_ds", "label": label, "synthetic": True,
        "source": "unit test", "sampling_rate_hz": 101.0,
        "columns": {
            "t": {"quantity": "time", "unit": "s", "frame": "W"},
            "force_x": {"quantity": "force_x", "unit": unit, "frame": "load_cell"},
        },
    }
    if extra_col:
        pass  # mystery column deliberately undeclared
    (d / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return d, t, fx


def test_importer_unit_conversion_and_labels(tmp_path):
    d, t, fx = _write_dataset(tmp_path, unit="mN")
    ds = load_dataset(d)
    assert ds.label == "calibration" and ds.synthetic
    df = ds.to_dataframe()
    np.testing.assert_allclose(df["force_x"].to_numpy(), fx * 1e-3, rtol=1e-12)


def test_importer_rejects_undeclared_column(tmp_path):
    d, _, _ = _write_dataset(tmp_path, extra_col=True)
    with pytest.raises(ValueError, match="without quantity/unit/frame"):
        load_dataset(d)


def test_importer_rejects_unknown_unit(tmp_path):
    d, _, _ = _write_dataset(tmp_path, unit="furlongs")
    with pytest.raises(ValueError, match="unknown unit"):
        load_dataset(d)


def test_importer_validation_label(tmp_path):
    d, _, _ = _write_dataset(tmp_path, label="validation")
    assert load_dataset(d).label == "validation"


def test_alignment_recovers_known_phase_offset():
    f = 13.0
    t = np.linspace(0, 2.0, 20001)
    s = np.sin(2 * np.pi * f * t)
    tau = 0.0123  # known offset, s
    e = np.sin(2 * np.pi * f * (t - tau)) + 0.01 * np.random.default_rng(0).normal(size=len(t))
    assert abs(dominant_frequency(t, s) - f) < 0.1
    offset, drift = estimate_offset(t, s, t, e, f)
    assert abs(offset - (-tau)) < 0.001 * (1.0 / f)  # < 0.1% of a wingbeat
    assert abs(drift) < 1e-3


def test_phase_bin_average():
    T = 0.1
    t = np.arange(0, 10 * T, T / 400)
    x = np.sin(2 * np.pi * t / T)
    centres, prof = resample_to_phase(t, x, T, n_bins=400)
    np.testing.assert_allclose(prof, np.sin(2 * np.pi * centres), atol=0.02)


def test_filter_recovers_known_component():
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    slow = np.sin(2 * np.pi * 5 * t)
    x = slow + 0.5 * np.sin(2 * np.pi * 60 * t)
    y = apply_filter(t, x, FilterSpec(order=4, cutoff_hz=20.0,
                                      rationale="unit test: split 5 Hz from 60 Hz"))
    k = len(t) // 10  # trim filtfilt's edge transient (documented)
    err = np.abs(y[k:-k] - slow[k:-k]).max()
    assert err < 0.005


def test_metrics_hand_computed():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0, 1.0])
    assert rmse(a, b) == np.sqrt((0 + 1 + 4) / 3)
    assert mae(a, b) == 1.0  # mean of |0,1,2|
    assert nrmse(a, b) == rmse(a, b) / (b.max() - b.min()) if b.max() != b.min() else True
    assert abs(pearson(a, a) - 1.0) < 1e-15
    t = np.linspace(0, 1.0, 10001)
    f = 10.0
    x1 = np.sin(2 * np.pi * f * t)
    x2 = np.sin(2 * np.pi * f * (t - 0.25 / f))  # quarter period behind
    assert abs(phase_lag(t, x1, x2, f) - np.pi / 2) < 0.01
    # welch: use a 1 kHz grid so the 10 Hz peak is resolvable (nperseg 1024)
    t_w = np.arange(0, 2.0, 1.0 / 1000.0)
    xw = np.sin(2 * np.pi * f * t_w)
    xw2 = np.sin(2 * np.pi * f * (t_w - 0.25 / f))
    fr, pa, pb = welch_spectra(t_w, xw, xw2, nperseg=1024)
    assert fr[np.argmax(pa)] == pytest.approx(f, abs=1.0)


def test_nrmse_zero_range_rejected():
    with pytest.raises(ValueError, match="zero range"):
        nrmse(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
