"""Zero-phase filtering with a recorded spec (M4 Task 3).

scipy.signal.filtfilt with a FilterSpec stored in dataset metadata and the
comparison report; the same spec applies to simulated and experimental series
where the comparison requires it (brief §5.9 item 3).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import butter, filtfilt


class FilterSpec(BaseModel):
    """Documented filter configuration (recorded everywhere it is applied)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lowpass"] = "lowpass"
    order: int = Field(gt=0, default=4)
    cutoff_hz: float = Field(gt=0)
    # e.g. the AirPulse rig used 4th-order Butterworth, 50 Hz cutoff
    rationale: str | None = None


def apply_filter(
    t: np.ndarray, x: np.ndarray, spec: FilterSpec
) -> np.ndarray:
    """Zero-phase Butterworth filter; requires roughly uniform sampling."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    if spec.cutoff_hz >= fs / 2:
        raise ValueError(f"cutoff {spec.cutoff_hz} Hz >= Nyquist {fs / 2:.1f} Hz")
    b, a = butter(spec.order, spec.cutoff_hz / (fs / 2))
    if len(x) <= 3 * max(len(a), len(b)):
        raise ValueError(f"signal too short ({len(x)}) for filter order {spec.order}")
    # filtfilt's reflection padding leaves an edge transient of ~padlen samples;
    # comparisons should trim or tolerate the edges (documented in reports)
    return filtfilt(b, a, x)
