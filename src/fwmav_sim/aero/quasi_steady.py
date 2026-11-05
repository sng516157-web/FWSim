"""Extended quasi-steady blade-element aerodynamic model (M1 Task 5).

Force per blade element (brief §5.1):

    F_wing = sum_i (F_trans,i + F_rot,i + F_am,i)

Terms and sources (brief §16: cited, not from memory):

- **Translational** (Dickinson et al. 1999, Science 284:1954, Robofly fits;
  coefficients live in the vehicle YAML with provenance=literature):
      C_L(alpha) = cl_0 + cl_slope * sin(cl_freq * alpha_deg - cl_phase_deg)
      C_D(alpha) = cd_0 - cd_slope * cos(cd_freq * alpha_deg - cd_phase_deg)
  with alpha in degrees. The fits are tabulated for alpha in [0, 180 deg];
  negative angles use the flat-plate symmetry C_L(-a) = -C_L(a),
  C_D(-a) = C_D(a). Magnitudes q = 1/2 rho |v_rel|^2 c dr.
  Directions: drag opposes the element's motion relative to the air
  (F_D = -q C_D v_hat_rel); lift is perpendicular to v_rel in the
  chord-normal plane, on the -n_phys side for C_L > 0 (lift is -z in hover;
  frames.md §1.1). e_L = -normalize(n_phys - (n_phys . v_hat) v_hat).
- **Rotational circulation** (Sane & Dickinson 2002, J. Exp. Biol. 205,
  translational-rotational coupling term):
      dF_rot = rho |v_rel| Gamma_rot dr,  Gamma_rot = pi c^2 theta_w_dot (3/4 - x_hat_0)
  directed along e_L. Sign convention (frozen with the hand-computed unit
  test): theta_w_dot > 0 (leading edge pitching up, frames.md §3) during
  forward translation increases force toward -n_phys (lift side).
  Mirror note: theta_w_dot is the *commanded* pitch rate, identical for both
  wings (frames.md §4); e_L mirrors, so F_rot mirrors correctly.
- **Added mass** (flat plate per unit span, Sedov / Ellington lineage per the
  M1 stage file): apparent mass rho pi c^2 / 4 dr opposing the element's
  normal acceleration relative to the (uniform, non-accelerating) air:
      dF_am = -(rho pi c^2 / 4) dr * (a_element . n_phys) * n_phys

Not captured (brief §5.1, assumptions A-002, A-003): wake capture,
clap-and-fling, arbitrary separated flow, circulatory lag.

Degenerate case: when v_rel is parallel to n_phys (|alpha| = 90 deg exactly),
the lift direction is undefined; the lift and rotational directions are set to
zero (C_L(90 deg) is small in the fit). Documented in the M1 report.

Known issue fixed post-M1 (DR-003 / A-008): the fit is now evaluated on the
flipped-wing-equivalent incidence in [0, 90 deg] — before this fix, the
backstroke evaluated the fit at alpha = 135 deg and produced downforce.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.interface import AerodynamicModel, ElementForces, Environment
from fwmav_sim.kinematics.blade_element import ElementState
from fwmav_sim.vehicle.schema import AeroModelSpec

#: Coefficients that must be present (with provenance) in the YAML.
REQUIRED_COEFFICIENTS = (
    "cl_0", "cl_slope", "cl_freq", "cl_phase_deg",
    "cd_0", "cd_slope", "cd_freq", "cd_phase_deg",
)


class QuasiSteadyBladeElement(AerodynamicModel):
    """Default extended quasi-steady blade-element model (brief §5.1)."""

    def __init__(self, spec: AeroModelSpec):
        if spec.name != "quasi_steady_blade_element":
            raise ValueError(f"unknown aero model {spec.name!r}")
        from fwmav_sim.aero.registry import resolve_coefficients

        coefficients, set_info = resolve_coefficients(spec)
        self._form = set_info["form"] if set_info else "dickinson1999"
        if self._form == "dickinson1999":
            missing = [k for k in REQUIRED_COEFFICIENTS if k not in coefficients]
            if missing:
                raise ValueError(f"aero coefficients missing from YAML: {missing}")
        self._c = {k: p.value for k, p in coefficients.items()}
        #: registry metadata when the vehicle named a coefficient set (M6);
        #: None for inline coefficients
        self.set_info = set_info

    def cl_cd(self, alpha_rad: float) -> tuple[float, float]:
        """C_L (signed), C_D from the Dickinson 1999 fits.

        Fit-domain convention (DR-003, assumption A-008): the published fits
        cover alpha in [0, 180 deg], where alpha > 90 deg describes a
        *non-flipped* trailing-edge-first plate. A flapping wing flips at
        stroke reversal, presenting the same surface at the same incidence on
        both half-strokes, so the fits are evaluated on the folded incidence
        |alpha| in [0, 90 deg] with odd (C_L) / even (C_D) symmetry and the
        sign of the flight-convention alpha. Hover regression tests pin that
        both half-strokes then produce upward force.
        """
        c = self._c
        a_deg = np.degrees((alpha_rad + np.pi) % (2 * np.pi) - np.pi)
        sign = 1.0 if a_deg >= 0 else -1.0
        a = abs(a_deg)
        if a > 90.0:
            a = 180.0 - a
        if self._form == "poststall_classical":
            # classical post-stall flat plate (registry; exact evaluation)
            ar = np.radians(a)
            return sign * c["cn"] * np.sin(ar) ** 2 * np.cos(ar), c["cn"] * np.sin(ar) ** 3
        cl = c["cl_0"] + c["cl_slope"] * np.sin(np.radians(c["cl_freq"] * a - c["cl_phase_deg"]))
        cd = c["cd_0"] - c["cd_slope"] * np.cos(np.radians(c["cd_freq"] * a - c["cd_phase_deg"]))
        return sign * cl, cd

    def element_forces(self, state: ElementState, env: Environment) -> ElementForces:
        v_rel = state.v_W - env.v_air_W
        speed = float(np.linalg.norm(v_rel))
        zero = np.zeros(3)

        # physical wing normal: +z_BE (right) / -z_BE (left) (frames.md §4)
        n_phys = state.R_W_BE @ np.array([0.0, 0.0, 1.0 if state.side == "right" else -1.0])

        # added mass: depends on acceleration, not speed — evaluate always
        a_n = float(state.a_W @ n_phys)  # uniform/still air: a_air = 0 (A-006)
        m_app = env.rho * np.pi * state.chord**2 / 4.0 * (state.area / state.chord)
        F_am = -m_app * a_n * n_phys

        if speed < 1e-14:
            # zero relative velocity => zero translational and rotational force;
            # added mass (acceleration-driven) still applies (documented exception)
            return ElementForces(F_trans=zero, F_rot=zero, F_am=F_am)

        v_hat = v_rel / speed
        cl, cd = self.cl_cd(state.alpha)
        dr = state.area / state.chord
        q = 0.5 * env.rho * speed**2 * state.chord * dr

        n_perp = n_phys - (n_phys @ v_hat) * v_hat
        n_perp_norm = np.linalg.norm(n_perp)
        if n_perp_norm < 1e-12:
            e_L = zero  # |alpha| = 90 deg exactly: lift direction undefined
        else:
            e_L = -n_perp / n_perp_norm

        F_trans = q * (cl * e_L - cd * v_hat)

        gamma_rot = np.pi * state.chord**2 * state.pitch_rate * (0.75 - state.pitch_axis_fraction)
        F_rot = env.rho * speed * gamma_rot * dr * e_L

        return ElementForces(F_trans=F_trans, F_rot=F_rot, F_am=F_am)
