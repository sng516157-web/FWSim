"""Probe 3: base pitch rate + joint rate cross-coupling — RNEA vs Jacobian bias."""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "derivations")
from probe_bias import _vehicle, jacobian_bias  # noqa: E402

from fwmav_sim.core.transforms import quat_from_euler  # noqa: E402
from fwmav_sim.dynamics.spatial import ArticulatedModel  # noqa: E402

model = ArticulatedModel(_vehicle())
q = np.concatenate([[0.0, 0.0, 0.0], quat_from_euler(0.0, 0.3, 0.0), [0.2, 0.0]])

for tag, u in [
    ("omega_y only", np.array([0, 1.0, 0, 0, 0, 0, 0, 0])),
    ("alpha_dot only", np.array([0, 0, 0, 0, 0, 0, 1.0, 0])),
    ("omega_y + alpha_dot", np.array([0, 1.0, 0, 0, 0, 0, 1.0, 0])),
    ("v_Bx + alpha_dot", np.array([0, 0, 0, 1.0, 0, 0, 1.0, 0])),
]:
    bJ = jacobian_bias(q, u)
    bR = model.rnea(q, u, np.zeros(8), gravity=False)
    d = np.abs(bJ - bR).max()
    print(f"{tag:22s} max|bJ - bRNEA| = {d:.3e}")
    if d > 1e-10:
        print("  bJ:  ", bJ.round(10))
        print("  bRNEA:", bR.round(10))
