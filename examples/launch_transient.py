"""Launch transient: clamped on the rig, then released (DR-007).

The base constraint is experiment data — this experiment clamps the vehicle
for the first 0.1 s (you get the rig forces on the same plant), then releases
it into free flight. Watch the exact release instant in the printout.

Run:  python examples/launch_transient.py
"""

from pathlib import Path

import numpy as np

from fwmav_sim.simulation.run import dispatch_experiment

EXPERIMENT = Path("experiments/m6p75_launch_transient.yaml")


def main() -> None:
    ts, _, meta = dispatch_experiment(EXPERIMENT)
    release = meta["time_series_meta"]["release_time"] if "time_series_meta" in meta \
        else 0.1
    t, p = ts.t, ts["p"]
    pre = t < release
    print(f"clamped until t = {release} s, then free flight")
    print(f"max |position| while clamped: {np.abs(p[pre]).max():.2e} m (rig)")
    i_rel = int(np.argmax(t >= release))
    print(f"at release: p = [{p[i_rel, 0]:+.4f}, {p[i_rel, 1]:+.4f}, {p[i_rel, 2]:+.4f}] m")
    print(f"at end ({t[-1]:.2f} s): p = [{p[-1, 0]:+.4f}, {p[-1, 1]:+.4f}, "
          f"{p[-1, 2]:+.4f}] m")
    fz = ts["wing_right/F_aero_total"][:, 2]
    print(f"Fz (right wing) at clamp midpoint: {fz[len(fz[pre]) // 2]:+.4f} N, "
          f"at end: {fz[-1]:+.4f} N")
    print("validity:")
    for w in meta["validity"]["warnings"]:
        print("  !", w)


if __name__ == "__main__":
    main()
