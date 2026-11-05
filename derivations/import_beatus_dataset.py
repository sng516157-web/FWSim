"""Import the Beatus-lab flapping-wing rig dataset into FWSim dataset format.

Source: Hadar933/AdaptiveSpectrumLayer (ICLR 2025, Sharvit, Karl & Beatus),
ML/Data/{kinematics,forces}_clean.pt — their custom robotic wing rig:
153 experiments, 5 kHz, 3 wing Euler angles (pitch, yaw, roll per their
README) + 4 force-sensor channels. The .pt files are converted (never
committed raw — 36 MB of torch tensors; the converted CSV.gz files are the
dataset of record).

Column semantics from their README; force units assumed Newton (magnitudes
±0.35 N are consistent with their wing scale) — flagged in the manifest for
confirmation against the paper. Label: validation (this data must never be
used for fitting — M4 importer rule).

Run: python derivations/import_beatus_dataset.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fwmav_sim.datasets.pt_loader import load_pt

SRC = Path("C:/Users/sng51/AppData/Local/Temp/beatus_repo/ML/Data")
OUT = Path("datasets/beatus_flapper")
RATE_HZ = 5000.0

MANIFEST = """\
name: beatus_flapper_rig
label: validation
sampling_rate_hz: 5000.0
source: "Sharvit, Karl & Beatus, ICLR 2025 (AdaptiveSpectrumLayer repo, ML/Data/*_clean.pt)"
provenance: "converted from the authors' public tensors; raw .pt not committed"
notes: >
  Custom robotic flapping-wing rig (single wing). Kinematics: 3 wing Euler
  angles (pitch, yaw, roll per the repo README), converted deg -> rad.
  Forces: 4 raw force-sensor channels, unit assumed N pending confirmation
  against the paper (magnitudes ~0.35 N consistent with the rig scale).
columns:
  t: {quantity: time, unit: s, frame: W}
  pitch: {quantity: wing_angle, unit: rad, frame: wing_root, uncertainty: "sensor/tracking noise (unquantified)"}
  yaw: {quantity: wing_angle, unit: rad, frame: wing_root}
  roll: {quantity: wing_angle, unit: rad, frame: wing_root}
  F1: {quantity: force, unit: N, frame: load_cell_1, note: "raw sensor channel; mapping to axes per the paper's Fig. 1"}
  F2: {quantity: force, unit: N, frame: load_cell_2, note: "raw sensor channel"}
  F3: {quantity: force, unit: N, frame: load_cell_3, note: "raw sensor channel"}
  F4: {quantity: force, unit: N, frame: load_cell_4, note: "raw sensor channel"}
"""


def main() -> None:
    kin = load_pt(SRC / "kinematics_clean.pt")
    forces = load_pt(SRC / "forces_clean.pt")
    assert len(kin) == len(forces) == 153
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.yaml").write_text(MANIFEST, encoding="utf-8")
    for i, (k, f) in enumerate(zip(kin, forces)):
        k, f = np.asarray(k), np.asarray(f)
        assert len(k) == len(f)
        t = np.arange(len(k)) / RATE_HZ
        arr = np.column_stack([t, np.radians(k), f])
        header = "t,pitch,yaw,roll,F1,F2,F3,F4"
        with gzip.open(OUT / f"exp_{i:03d}.csv.gz", "wt") as fh:
            np.savetxt(fh, arr, delimiter=",", header=header, comments="")
    print(f"wrote {len(kin)} experiments to {OUT}")


if __name__ == "__main__":
    main()
