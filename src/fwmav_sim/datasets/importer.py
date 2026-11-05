"""Experimental dataset importer (M4 Task 1).

A dataset is a directory with:
  - manifest.yaml: label (calibration | validation), per-column
    quantity/unit/frame, sampling rate, source/provenance, optional
    uncertainty, optional frame conversion note
  - one or more CSV files with a time column

Rules (brief §5.8, §5.9, §3.5, §11.4):
- every column must declare quantity, unit and frame; unlabeled or unit-less
  columns are rejected;
- conversion to SI happens at the boundary and is recorded in metadata;
- a dataset used for fitting can never afterwards carry the validation label
  (enforced by the manifest label being load-time immutable metadata).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

#: unit conversion factors to SI (length m, mass kg, time s, force N, angles rad)
_UNIT_TO_SI = {
    "m": 1.0, "mm": 1e-3, "cm": 1e-2,
    "g": 1e-3, "kg": 1.0,
    "s": 1.0, "ms": 1e-3,
    "N": 1.0, "mN": 1e-3,
    "rad": 1.0, "deg": np.pi / 180.0,
    "Hz": 1.0,
    "m/s": 1.0, "mm/s": 1e-3,
    "g mm^2": 1e-9, "kg m^2": 1.0,
    "N m": 1.0, "mN m": 1e-3,
    "V": 1.0,  # sensor voltages pass through with the unit recorded
    "1": 1.0,
}


class QuantitySpec(BaseModel):
    """One data column: physical quantity, declared unit and frame."""

    model_config = ConfigDict(extra="forbid")

    quantity: str = Field(min_length=1)  # e.g. "force_x", "pitch", "time"
    unit: str = Field(min_length=1)
    frame: str = Field(min_length=1)  # e.g. "W", "B", "load_cell", "paper_FLU"
    uncertainty: str | None = None
    note: str | None = None  # e.g. frame conversion applied


class DatasetManifest(BaseModel):
    """Manifest beside each dataset (brief §5.8 provenance)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    label: Literal["calibration", "validation"]
    synthetic: bool = False
    source: str = Field(min_length=1)  # e.g. "Leng rig 2024" or "generated from record X"
    provenance: str | None = None
    sampling_rate_hz: float | None = None
    time_column: str = "t"
    columns: dict[str, QuantitySpec]
    notes: str | None = None


class ExperimentalDataset(BaseModel):
    """Loaded dataset: SI-converted dataframe + manifest."""

    label: Literal["calibration", "validation"]
    synthetic: bool
    source: str
    manifest: DatasetManifest

    _df: pd.DataFrame | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dataframe(self) -> pd.DataFrame:
        assert self._df is not None
        return self._df

    def attach(self, df: pd.DataFrame) -> None:
        self._df = df


def load_dataset(dataset_dir: str | Path) -> ExperimentalDataset:
    """Load manifest + CSV(s), convert to SI, record conversions."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.yaml in {dataset_dir}")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )

    csvs = sorted(dataset_dir.glob("*.csv")) + sorted(dataset_dir.glob("*.csv.gz"))
    if not csvs:
        raise FileNotFoundError(f"no CSV files in {dataset_dir}")
    # per-file identity is preserved (multi-experiment datasets like the
    # Beatus rig set): source_file column carries the CSV stem
    frames = []
    for c in csvs:
        df_c = pd.read_csv(c)
        if len(csvs) > 1:
            df_c["source_file"] = c.stem
        frames.append(df_c)
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in manifest.columns if c not in df.columns]
    if missing:
        raise ValueError(f"manifest columns missing from CSV: {missing}")
    undeclared = [c for c in df.columns if c not in manifest.columns and c != "source_file"]
    if undeclared:
        raise ValueError(f"CSV columns without quantity/unit/frame declaration: {undeclared}")

    conversions: dict[str, str] = {}
    for col, spec in manifest.columns.items():
        if spec.unit not in _UNIT_TO_SI:
            raise ValueError(f"unknown unit {spec.unit!r} for column {col!r} "
                             f"(add it to _UNIT_TO_SI with its SI factor)")
        factor = _UNIT_TO_SI[spec.unit]
        if factor != 1.0:
            df[col] = df[col] * factor
            conversions[col] = f"{spec.unit} -> SI (x{factor})"
    if manifest.time_column not in df.columns:
        raise ValueError(f"time column {manifest.time_column!r} not in CSV")
    df = df.set_index(manifest.time_column).sort_index()

    ds = ExperimentalDataset(label=manifest.label, synthetic=manifest.synthetic,
                             source=manifest.source, manifest=manifest)
    ds.attach(df)
    ds.__dict__["conversions"] = conversions
    return ds
