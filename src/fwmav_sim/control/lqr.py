"""Continuous-time LQR over the linearized reduced model (M6 Task 8)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """K such that u = -K x solves the continuous-time LQR problem."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)
