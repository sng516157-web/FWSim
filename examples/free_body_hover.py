"""Free-body hover: the MAV actually flies.

Same vehicle and kinematics as the fixed-body example, but now the 6-DoF
body is released: the articulated dynamics (thorax + abdomen) integrate
under the time-resolved aerodynamic forces. Watch whether it holds hover,
drifts, or tumbles — with placeholder (estimated) parameters it will NOT
trim, and that is the point of the honesty layer.

Run:  python examples/free_body_hover.py
"""

from pathlib import Path

import numpy as np

from fwmav_sim.simulation.run import dispatch_experiment

EXPERIMENT = Path("experiments/m3_free_body_hover.yaml")


def main() -> None:
    ts, vehicle_path, meta = dispatch_experiment(EXPERIMENT)

    p = ts["p"]  # body position in the world frame (z down)
    euler_z = ts["p"][:, 2]
    print(f"status: {meta['status']}  ({len(ts.t)} samples over "
          f"{ts.t[-1]:.2f} s)")
    print(f"start p = [{p[0, 0]:+.4f}, {p[0, 1]:+.4f}, {p[0, 2]:+.4f}] m")
    print(f"end   p = [{p[-1, 0]:+.4f}, {p[-1, 1]:+.4f}, {p[-1, 2]:+.4f}] m")
    print(f"peak sink rate = {np.diff(euler_z).max() / (ts.t[1] - ts.t[0]):.3f} m/s")
    print(f"record meta says: abdomen_mode={ts.meta['abdomen_mode']}, "
          f"assumptions={ts.meta['assumptions']}")
    print("note: with estimated placeholder parameters the vehicle is not "
          "trimmed — drifting/falling is expected, not a bug. Validity:")
    for w in meta["validity"]["warnings"]:
        print("  !", w)


if __name__ == "__main__":
    main()
