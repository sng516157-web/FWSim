"""Fixed-body hover: the simplest complete FWSim run.

A fixed-body experiment clamps the vehicle on a force rig (think wind-tunnel
balance), flaps the wings with prescribed kinematics, and records the
aerodynamic forces. This is the "hello world" of the project: one vehicle
YAML, one experiment YAML, one command.

Run:  python examples/fixed_body_hover.py
"""

from pathlib import Path

from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.simulation.run import dispatch_experiment

EXPERIMENT = Path("experiments/m1_fixed_body_hover.yaml")


def main() -> None:
    # dispatch_experiment runs the experiment in process (no files written);
    # `fwmav-sim run <same file>` additionally saves a record to outputs/.
    ts, vehicle_path, meta = dispatch_experiment(EXPERIMENT)

    period = ts.meta["wingbeat_period_s"]
    print(f"vehicle: {vehicle_path}")
    print(f"kind: {ts.meta['kind']}, {ts.meta['n_wingbeats']} wingbeats at "
          f"{ts.meta['flap_frequency_hz']} Hz")
    for wing in ("wing_right", "wing_left"):
        mean, spread = wingbeat_mean(ts, f"{wing}/F_total", period)
        print(f"{wing}: cycle-mean force [N] = "
              f"[{mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f}]  "
          f"(cycle spread {spread.max():.2e})")
    print("note: z is DOWN, so a negative Fz is lift. Validity warnings:")
    for w in meta["validity"]["warnings"]:
        print("  !", w)


if __name__ == "__main__":
    main()
