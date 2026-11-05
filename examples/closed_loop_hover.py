"""Closed-loop hover: trim -> linearize -> LQR -> simulate.

The full M6 design loop on the Level-1 reduced model: solve the hover trim
(wing placement included), linearize the nonlinear plant at the trim, design
an LQR, and regulate back from a perturbed initial state. The plant includes
the cycle-averaged aerodynamic body damping (flapping counter-torque) —
without it the trim is open-loop unstable.

Run:  python examples/closed_loop_hover.py
"""

from pathlib import Path

import numpy as np

from fwmav_sim.simulation.run import dispatch_experiment

EXPERIMENT = Path("experiments/m6_closed_loop_hover.yaml")


def main() -> None:
    ts, vehicle_path, meta = dispatch_experiment(EXPERIMENT)

    trim = meta["trim"]
    print("trim solution:")
    print(f"  thrust scale      = {trim['thrust_scale']:.3f}")
    print(f"  alpha_abd (rad)   = {trim['alpha_abd_trim']:.4f}  (bounded to joint limits)")
    print(f"  theta_trim (rad)  = {trim['theta_trim']:.4f}")
    print(f"  x_T_shift (m)     = {trim['x_T_shift']:.4f}  (wing placement!)")
    print(f"  residuals         = {np.abs(trim['residual']).max():.2e}")
    print(f"closed loop from perturbation {meta['experiment_file'].split('/')[-1]}:")
    print(f"  theta: {ts['theta'][0]:+.3f} -> {ts['theta'][-1] - trim['theta_trim']:+.4f} rad")
    print(f"  z:     {ts['z'][0]:+.4f} -> {ts['z'][-1]:+.4f} m")
    print(f"  max |du_thrust|  = {np.abs(ts['du_thrust']).max():.3f} (saturation 0.5)")
    print(f"  max |du_tau|     = {np.abs(ts['du_tau']).max():.2e} N m")
    print("validity:")
    for w in meta["validity"]["warnings"]:
        print("  !", w)


if __name__ == "__main__":
    main()
