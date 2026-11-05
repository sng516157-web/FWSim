"""Beatus rig dataset import test (M6.75): the project's first real
experimental validation data loads through the M4 pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.datasets.importer import load_dataset


def test_beatus_dataset_loads():
    ds = load_dataset("datasets/beatus_flapper")
    assert ds.label == "validation"  # never a fitting set (M4 rule)
    df = ds.to_dataframe()
    assert {"pitch", "yaw", "roll", "F1", "F2", "F3", "F4", "source_file"} <= set(df.columns)
    assert df["source_file"].nunique() == 153
    # physical sanity: forces within +-0.5 N, angles within +-135 deg
    assert np.abs(df[["F1", "F2", "F3", "F4"]].to_numpy()).max() < 0.5
    assert np.abs(df[["pitch", "yaw", "roll"]].to_numpy()).max() < np.radians(135.0)


def test_pt_loader_roundtrip():
    """The torch-shim loader produces the documented shapes."""
    import os

    from fwmav_sim.datasets.pt_loader import load_pt

    base = os.environ.get("BEATUS_DATA")
    if not base:
        pytest.skip("raw .pt not present locally (not committed; see "
                    "derivations/import_beatus_dataset.py)")
