"""Time-series container shared by simulation output, analysis and viz.

A TimeSeries holds a time vector plus named channels whose first axis is
time. Wingbeat averaging (analysis/averaging.py) is always derived from these
time-resolved arrays, never recomputed independently (brief §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeSeries:
    """Named time-series channels on a common time base."""

    t: np.ndarray  # (N,) time, s
    channels: dict[str, np.ndarray] = field(default_factory=dict)  # first axis = time
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        for name, data in self.channels.items():
            arr = np.asarray(data, dtype=float)
            if arr.shape[0] != self.t.shape[0]:
                raise ValueError(
                    f"channel {name!r} has {arr.shape[0]} samples, t has {self.t.shape[0]}"
                )
            self.channels[name] = arr

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    def channel_names(self) -> list[str]:
        return list(self.channels)
