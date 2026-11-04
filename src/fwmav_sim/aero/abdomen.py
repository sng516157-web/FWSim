"""Level-3 abdomen aerodynamics (brief §6.5; post-M5 item 4).

The abdomen is modelled as a flat plate with its own centre of pressure
(AbdomenAeroSpec: area, cp_offset behind the joint along the abdomen axis,
own Dickinson-style CL/CD fit coefficients — same functional form as the wing
fits, quasi_steady.py, but a separate coefficient set because abdomen Reynolds
numbers differ from wing Re).

Force law at the centre of pressure:

    v_rel  = v_cop - v_air          (spanwise (y_abd) component excluded)
    alpha  = atan2(v_rel . n, v_rel . c)   (c = abdomen axis, n = plate normal)
    F      = 0.5 rho |v_rel|^2 S [ C_L(alpha) e_L - C_D(alpha) v_hat_rel ]

with e_L the lift direction in the plate's chord-normal plane (same
convention as quasi_steady.py; DR-003 alpha folding applies via the shared
cl_cd evaluation).

Assumptions: A-002/A-003 (quasi-steady, no wake interaction with the wings —
documented limitation), A-006 (uniform air). The wrench is applied to the
abdomen body at the CoP through the model's body Jacobians (f_ext), so the
abdomen joint and thorax feel the correct reaction.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import REQUIRED_COEFFICIENTS, QuasiSteadyBladeElement
from fwmav_sim.vehicle.schema import AbdomenAeroSpec, AeroModelSpec


class AbdomenPlateAero:
    """Flat-plate abdomen aerodynamics; see module docstring."""

    def __init__(self, spec: AbdomenAeroSpec):
        missing = [k for k in REQUIRED_COEFFICIENTS if k not in spec.coefficients]
        if missing:
            raise ValueError(f"abdomen aero coefficients missing from YAML: {missing}")
        self.area = spec.area.value
        self.cp_offset = spec.cp_offset.value
        # reuse the wing model's fit evaluation (DR-003 folding included)
        self._fit = QuasiSteadyBladeElement(
            AeroModelSpec(name="quasi_steady_blade_element",
                          coefficients=spec.coefficients))

    def force(self, v_cop_W: np.ndarray, R_W_abd: np.ndarray, env: Environment
              ) -> tuple[np.ndarray, float]:
        """(F_W, alpha) at the centre of pressure.

        R_W_abd maps abdomen-body coords -> world. The abdomen axis is -x_abd
        (extends backward from the joint, reduced_model_spec.md §4); the plate
        normal is +z_abd; y_abd (roll axis) is the spanwise direction whose
        flow component is excluded (blade-element convention).
        """
        v_rel = np.asarray(v_cop_W, dtype=float) - env.v_air_W
        c_dir = R_W_abd @ np.array([-1.0, 0.0, 0.0])
        n_dir = R_W_abd @ np.array([0.0, 0.0, 1.0])
        y_dir = R_W_abd @ np.array([0.0, 1.0, 0.0])
        v_plane = v_rel - float(v_rel @ y_dir) * y_dir
        speed = float(np.linalg.norm(v_plane))
        if speed < 1e-14:
            return np.zeros(3), 0.0
        alpha = float(np.arctan2(v_plane @ n_dir, v_plane @ c_dir))
        v_hat = v_plane / speed
        cl, cd = self._fit.cl_cd(alpha)
        n_perp = n_dir - float(n_dir @ v_hat) * v_hat
        n_perp_norm = float(np.linalg.norm(n_perp))
        e_L = np.zeros(3) if n_perp_norm < 1e-12 else -n_perp / n_perp_norm
        q = 0.5 * env.rho * speed**2 * self.area
        return q * (cl * e_L - cd * v_hat), alpha
