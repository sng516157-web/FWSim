"""Free-body coupled simulation (M3 Tasks 4-7).

The plant: ArticulatedModel (6-DoF thorax + abdomen pitch/roll) + M1
quasi-steady aero coupled through the element kinematics (brief §5.4) +
abdomen control (locked / prescribed / profile-limited / torque-driven) +
mechanism-isolation switches (brief §6.6).

Force mapping: each blade element's aerodynamic force is applied to the
thorax as a spatial force (wings are rigidly attached); the generalized force
is J^T F via the thorax wrench (brief §5.4); per-term forces remain
inspectable.

Wing structural mass (post-M5, DR-006 / A-010): wings with ``mass`` set in
the vehicle YAML are kinematically driven rigid appendages — their spatial
inertia folds into the base block of M(q) and their velocity/prescribed/
gravity reaction wrench joins the right-hand side
(dynamics/wing_inertia.py). Wings without a mass stay massless (the M1-M5
default). Switchable via IsolationSpec.wing_inertia_enabled (brief §6.6).

Assumption notes: A-004 (prescribed wing kinematics), A-009 (fluid added-mass
term excludes body accelerations), A-010 (wing mass as uniform thin plate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fwmav_sim.actuation.prescribed_abdomen import AbdomenProfile, prescribed_alpha
from fwmav_sim.actuation.profile_limited import ProfileLimiter
from fwmav_sim.actuation.torque_driven import TorqueActuator
from fwmav_sim.aero.interface import AerodynamicModel, Environment
from fwmav_sim.core.transforms import cross3, quat_normalize, quat_to_matrix
from fwmav_sim.dynamics.spatial import ArticulatedModel
from fwmav_sim.dynamics.wing_inertia import (
    wing_inertial_state,
    wing_inertial_terms,
    wing_mass_properties,
)
from fwmav_sim.kinematics.blade_element import discretize
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import VehicleDefinition


@dataclass(frozen=True)
class IsolationSpec:
    """Mechanism-isolation switches (brief §6.6)."""

    mass_shift_enabled: bool = True
    inertial_coupling_enabled: bool = True
    abdomen_aero_enabled: bool = False  # stub: raises until the Level-3 model exists
    wing_inertia_enabled: bool = True  # wing structural mass (DR-006); no-op if massless

    def __post_init__(self):
        if self.abdomen_aero_enabled:
            raise NotImplementedError(
                "abdomen aerodynamics is a Level-3 capability (brief §6.5): "
                "the flag exists but no model is implemented yet"
            )


@dataclass
class AbdomenControl:
    """Abdomen control selection for the free-body plant."""

    mode: Literal["locked", "prescribed", "profile_limited", "torque_driven"] = "locked"
    profile: AbdomenProfile | None = None
    limiter: ProfileLimiter | None = None
    actuator: TorqueActuator | None = None
    limits: tuple[float, float] | None = None
    joint_damping: float = 0.0  # c_a, N m s / rad (applied on free joints)


class FreeBodyPlant:
    """Everything between the integrator and the physics models."""

    def __init__(
        self,
        vehicle: VehicleDefinition,
        kinematics: PrescribedWingKinematics,
        env: Environment,
        aero_model: AerodynamicModel | None,
        control: AbdomenControl | None = None,
        isolation: IsolationSpec = IsolationSpec(),
    ):
        self.model = ArticulatedModel(vehicle)
        self.vehicle = vehicle
        if isinstance(kinematics, dict):
            self.kinematics_map = kinematics
            self.kinematics = next(iter(kinematics.values()))
        else:
            self.kinematics_map = {w.name: kinematics for w in vehicle.wings}
            self.kinematics = kinematics
        self.env = env
        self.aero = aero_model
        self.control = control or AbdomenControl()
        self.isolation = isolation
        self._elements = {
            w.name: discretize(w, vehicle.blade_elements.n_elements,
                               vehicle.blade_elements.discretization)
            for w in vehicle.wings
        }
        self.nj = self.model.nj
        # wing structural mass (DR-006): precompute planform mass properties
        # for wings that carry a mass; None entries stay massless
        self._wing_mass = {
            w.name: wing_mass_properties(w) for w in vehicle.wings
        }
        # extra plant states: profile limiter [alpha_lim, alpha_dot_lim]
        self.n_extra = 2 if self.control.mode == "profile_limited" else 0
        self.ny = 7 + self.nj + 6 + self.nj + self.n_extra
        # recorded per-call diagnostics (sampled by the runner)
        self.last_reaction = np.zeros(self.nj)
        self.last_alpha_cmd = 0.0
        self.last_saturated = False
        self.last_F_wings: dict[str, np.ndarray] = {}
        self.last_lever_arms: dict[str, np.ndarray] = {}
        self.last_F_wing_inertia: dict[str, np.ndarray] = {}

    # --- force assembly -----------------------------------------------------

    def _aero_wrench(self, t: float, qm: np.ndarray, u: np.ndarray):
        """Aero forces on the thorax + per-wing records. Returns
        (f_ext dict for the model, record dict)."""
        if self.aero is None:
            return None, {}
        p, quat = qm[:3], qm[3:7]
        R_wb = quat_to_matrix(quat_normalize(quat))
        omega_B, v_B = u[:3], u[3:6]
        omega_W = R_wb @ omega_B
        v_W = R_wb @ v_B

        f_thorax = np.zeros(6)
        rec: dict[str, dict[str, np.ndarray]] = {}
        from fwmav_sim.kinematics.blade_element import WingFrameContext, element_state_ctx

        for wing in self.vehicle.wings:
            ctx = WingFrameContext(wing, self.kinematics_map[wing.name], t, R_wb, p,
                                   omega_W, v_W, self.env.v_air_W)
            F_sum = {"F_trans": np.zeros(3), "F_rot": np.zeros(3), "F_am": np.zeros(3)}
            r_app = np.zeros(3)  # force-weighted application point (documented)
            w_tot = 0.0
            for el in self._elements[wing.name]:
                st = element_state_ctx(wing, el, ctx, nu=self.env.nu)
                ef = self.aero.element_forces(st, self.env)
                for key in F_sum:
                    F_sum[key] += getattr(ef, key)
                F_el = ef.resultant
                w_tot += np.linalg.norm(F_el)
                r_app += np.linalg.norm(F_el) * st.r_W
                f_B = R_wb.T @ F_el
                r_el_B = R_wb.T @ (st.r_W - p)
                f_thorax += np.concatenate([cross3(r_el_B, f_B), f_B])
            r_app = r_app / w_tot if w_tot > 1e-15 else np.zeros(3)
            rec[wing.name] = {**F_sum, "r_app_W": r_app}
        return {"thorax": f_thorax}, rec

    # --- plant rhs ------------------------------------------------------------

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        nj = self.nj
        p, quat, qj = y[:3], y[3:7], y[7:7 + nj]
        v_W = y[7 + nj:10 + nj]
        w_B = y[10 + nj:13 + nj]
        qd_j = y[13 + nj:13 + 2 * nj]
        extra = y[13 + 2 * nj:]
        quat = quat_normalize(quat)
        R_wb = quat_to_matrix(quat)
        qm = np.concatenate([p, quat, qj])
        u = np.concatenate([w_B, R_wb.T @ v_W, qd_j])

        # --- abdomen control
        ctl = self.control
        prescribed: dict[int, tuple[float, float]] = {}
        tau = np.zeros(self.model.nv)
        extra_rhs = np.zeros(0)
        alpha_cmd = qj[0] if nj else 0.0
        if ctl.mode == "locked":
            prescribed = {j: (0.0, 0.0) for j in range(nj)}
        elif ctl.mode == "prescribed":
            a, ad, add = prescribed_alpha(ctl.profile, t, ctl.limits)
            alpha_cmd = a
            qm[7] = a
            u[6] = ad
            prescribed[0] = (ad, add)
            if nj > 1:
                prescribed[1] = (0.0, 0.0)  # roll locked in M3 experiments
        elif ctl.mode == "profile_limited":
            lim_state = extra[:2]
            a_cmd = prescribed_alpha(ctl.profile, t, ctl.limits)[0]
            alpha_cmd = a_cmd
            lim_rhs = ctl.limiter.rhs(lim_state, t, ctl.profile)
            extra_rhs = lim_rhs
            a_lim, ad_lim = lim_state
            add_lim = lim_rhs[1]
            qm[7] = a_lim
            u[6] = ad_lim
            prescribed[0] = (ad_lim, add_lim)
            if nj > 1:
                prescribed[1] = (0.0, 0.0)
        elif ctl.mode == "torque_driven":
            tau_j, sat = ctl.actuator.applied_torque(t, qj[0], qd_j[0])
            self.last_saturated = sat
            tau[6] = tau_j
            if nj > 1:
                tau[7] = -ctl.joint_damping * qd_j[1]
        self.last_alpha_cmd = alpha_cmd

        # --- isolation switches (brief §6.6)
        qm_eff = qm.copy()
        u_eff = u.copy()
        prescribed_eff = dict(prescribed)
        if not self.isolation.mass_shift_enabled:
            qm_eff[7:7 + nj] = 0.0  # CoM/inertia frozen at neutral
        if not self.isolation.inertial_coupling_enabled:
            # quasi-static CoG shift: joint rates/accelerations do not couple
            u_eff[6:6 + nj] = 0.0
            prescribed_eff = {j: (0.0, 0.0) for j in prescribed_eff}

        # --- aero + dynamics
        f_ext, rec = self._aero_wrench(t, qm_eff, u_eff)

        # --- wing structural mass (DR-006): fold wing spatial inertia into M
        # and the velocity/prescribed/gravity reaction into the rhs
        m_extra = None
        rhs_extra = None
        wing_states: list[tuple[str, object]] = []
        if self.isolation.wing_inertia_enabled and any(
                p is not None for p in self._wing_mass.values()):
            m_extra = np.zeros((self.model.nv, self.model.nv))
            rhs_extra = np.zeros(self.model.nv)
            for wing in self.vehicle.wings:
                props = self._wing_mass[wing.name]
                if props is None:
                    continue
                st = wing_inertial_state(wing, self.kinematics_map[wing.name], props,
                                         t, R_wb, p, w_B, u_eff[3:6])
                M_add, Q_bias = wing_inertial_terms(st)
                m_extra[:6, :6] += M_add
                rhs_extra[:6] += Q_bias
                wing_states.append((wing.name, st))

        ud, tau_full = self.model.forward_dynamics(
            qm_eff, u_eff, tau, f_ext=f_ext, prescribed_joints=prescribed_eff or None,
            m_extra=m_extra, rhs_extra=rhs_extra)
        self.last_reaction = tau_full[6:6 + nj]
        self.last_F_wings = rec
        self.last_F_wing_inertia = {
            name: st.wrench_on_thorax(ud[:3], ud[3:6])[0]
            for name, st in wing_states
        }

        # lever arms: force application point - instantaneous CoM (brief §10 M3)
        r_com = self.model.com(qm_eff)
        self.last_lever_arms = {w: r["r_app_W"] - r_com for w, r in rec.items()}

        vd_W = R_wb @ (ud[3:6] + cross3(w_B, R_wb.T @ v_W))
        from fwmav_sim.core.transforms import quat_rate
        qj_out = (np.array([prescribed[j][0] for j in sorted(prescribed)])
                  if prescribed else u[6:6 + nj])
        qdj_out = (np.array([prescribed[j][1] for j in sorted(prescribed)])
                   if prescribed else ud[6:6 + nj])
        out = np.concatenate([
            v_W,
            quat_rate(quat, w_B),
            qj_out,
            vd_W,
            ud[:3],
            qdj_out,
            extra_rhs,
        ])
        return out
