"""Wing structural mass tests (post-M5 item 1; DR-006, A-010).

- planform mass properties vs known plate results (all three planforms)
- chain angular acceleration vs finite difference of the chain angular velocity
- rigid-wing inertial terms (M_add, Q_bias) vs an independent point-mass
  assembly over blade elements (with per-element plate corrections)
- fixed-body inertial root force: cycle mean equals wing weight (fluctuating
  part zero-mean for periodic kinematics)
- free body with flapping massive wings, aero off: total linear and angular
  momentum (bodies + wings) follow the external gravity load exactly — the
  gold-standard check that the M(q)-folded wing inertia couples correctly
- backward compatibility: massless wings keep the M1-M5 behaviour
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.core.transforms import quat_from_euler, quat_normalize, quat_to_matrix
from fwmav_sim.dynamics.wing_inertia import (
    G_W,
    wing_inertial_state,
    wing_inertial_terms,
    wing_mass_properties,
)
from fwmav_sim.kinematics.blade_element import (
    discretize,
    element_state_free_body,
    wing_chain_with_alpha,
)
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics, wing_state
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated"}


def _param(v, unit):
    return {"value": v, "unit": unit, "provenance": _PROV}


def _wing(planform="rectangular", side="right", mass=0.005, span=0.30,
          c_r=0.10, c_t=0.05, f_pa=0.25, hinge=(0.01, 0.02, -0.005), dihedral=None):
    d = {
        "name": f"wing_{side}", "side": side,
        "span": _param(span, "m"),
        "root_chord": _param(c_r, "m"),
        "tip_chord": _param(c_t, "m"),
        "planform": planform,
        "hinge_location": {"value": list(hinge), "unit": "m", "provenance": _PROV},
        "pitch_axis_fraction": _param(f_pa, "1"),
    }
    if mass is not None:
        d["mass"] = _param(mass, "kg")
    if dihedral is not None:
        d["dihedral_angle"] = _param(dihedral, "rad")
    return d


def _vehicle(wings, mass=0.02):
    return VehicleDefinition.model_validate({
        "name": "wingmass_test",
        "bodies": [{
            "name": "body",
            "mass": _param(mass, "kg"),
            "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
            "inertia": {"value": (5e-5 * np.eye(3)).tolist(), "unit": "kg m^2",
                        "provenance": _PROV},
        }],
        "joints": [],
        "wings": wings,
        "aero": {"name": "none", "coefficients": {}},
    })


def _kin(f_hz=10.0, phi_amp=0.9, theta_amp=0.5, beta=0.0, psi_amp=None):
    d = {
        "flap_amplitude": _param(phi_amp, "rad"),
        "flap_frequency": _param(f_hz, "Hz"),
        "pitch_amplitude": _param(theta_amp, "rad"),
        "pitch_phase": _param(0.6, "rad"),
        "pitch_offset": _param(0.1, "rad"),
        "stroke_plane_angle": _param(beta, "rad"),
    }
    if psi_amp is not None:
        d["deviation_amplitude"] = _param(psi_amp, "rad")
        d["deviation_phase"] = _param(0.3, "rad")
    return PrescribedWingKinematics.model_validate(d)


# --- planform mass properties --------------------------------------------------

def test_rectangular_plate_properties():
    v = _vehicle([_wing("rectangular", mass=0.005, span=0.30, c_r=0.10, c_t=0.10,
                        f_pa=0.0)])
    p = wing_mass_properties(v.wings[0])
    m, R, c = 0.005, 0.30, 0.10
    assert p.mass == m
    # CoM: midspan; half a chord behind the leading-edge pitch axis
    np.testing.assert_allclose(p.r_com_WR, [-0.5 * c, 0.5 * R, 0.0], atol=1e-15)
    # rectangular plate: I_xx = m R^2/12, I_yy = m c^2/12, I_zz = sum
    np.testing.assert_allclose(
        np.diag(p.I_com_WR),
        [m * R**2 / 12, m * c**2 / 12, m * (R**2 + c**2) / 12], rtol=1e-12)


def test_trapezoidal_plate_properties():
    c_r, c_t, R, m = 0.10, 0.05, 0.30, 0.004
    v = _vehicle([_wing("trapezoidal", mass=m, span=R, c_r=c_r, c_t=c_t, f_pa=0.5)])
    p = wing_mass_properties(v.wings[0])
    area = R * (c_r + c_t) / 2
    s_com = (R**2 * (c_r / 2 + (c_t - c_r) / 3)) / area
    np.testing.assert_allclose(p.r_com_WR, [0.0, s_com, 0.0], atol=1e-15)
    # numeric quadrature cross-check of the full inertia
    s = np.linspace(0, R, 20001)
    c = c_r + (c_t - c_r) * s / R
    rho = m / area
    i_xx_q = rho * np.trapezoid((s - s_com) ** 2 * c, s)
    assert abs(p.I_com_WR[0, 0] - i_xx_q) < 1e-6 * p.I_com_WR[0, 0] + 1e-15


def test_elliptical_plate_properties():
    c_r, R, m = 0.10, 0.30, 0.003
    v = _vehicle([_wing("elliptical", mass=m, span=R, c_r=c_r, c_t=c_r, f_pa=0.25)])
    p = wing_mass_properties(v.wings[0])
    area = c_r * R * np.pi / 4
    s_com = (c_r * R**2 / 3) / area  # = 4R / (3 pi)
    np.testing.assert_allclose(p.r_com_WR[1], s_com, rtol=1e-12)
    np.testing.assert_allclose(p.r_com_WR[0],
                               (0.25 - 0.5) * (c_r**2 * R * 2 / 3) / area, rtol=1e-12)
    s = np.linspace(0, R, 20001)
    c = c_r * np.sqrt(np.clip(1 - (s / R) ** 2, 0, None))
    rho = m / area
    i_xx_q = rho * np.trapezoid((s - s_com) ** 2 * c, s)
    assert abs(p.I_com_WR[0, 0] - i_xx_q) < 1e-6 * p.I_com_WR[0, 0] + 1e-15


def test_massless_wing_default():
    v = _vehicle([_wing(mass=None)])
    assert wing_mass_properties(v.wings[0]) is None


def test_wing_mass_must_be_positive():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _vehicle([_wing(mass=-0.001)])


# --- chain angular acceleration -------------------------------------------------

def test_chain_alpha_matches_finite_difference():
    kin = _kin(psi_amp=0.2)
    for side in ("right", "left"):
        for t in (0.0, 0.013, 0.047):
            h = 1e-6
            _, _, _, _, al = wing_chain_with_alpha(wing_state(kin, t), side)
            _, _, _, om_p, _ = wing_chain_with_alpha(wing_state(kin, t + h), side)
            _, _, _, om_m, _ = wing_chain_with_alpha(wing_state(kin, t - h), side)
            np.testing.assert_allclose(al, (om_p - om_m) / (2 * h), rtol=1e-6, atol=1e-7)


# --- rigid-wing terms vs point-mass assembly -----------------------------------

def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _element_plate_inertia(m_el, chord, area, R_W_BE):
    """Thin rectangular plate element about its own centre, world coords."""
    dr = area / chord
    i_el = np.diag([m_el * dr**2 / 12, m_el * chord**2 / 12,
                    m_el * (dr**2 + chord**2) / 12])
    return R_W_BE @ i_el @ R_W_BE.T


def test_rigid_wing_matches_point_mass_mass_matrix():
    wing = _vehicle([_wing("trapezoidal", side="right", dihedral=0.4)]).wings[0]
    kin = _kin(beta=0.3, psi_amp=0.15)
    rng = np.random.default_rng(3)
    R_wb = quat_to_matrix(quat_from_euler(0.2, -0.3, 0.1))
    p_W = rng.uniform(-0.1, 0.1, 3)
    omega_B = rng.uniform(-2, 2, 3)
    v_B = rng.uniform(-1, 1, 3)
    props = wing_mass_properties(wing)
    st = wing_inertial_state(wing, kin, props, 0.037, R_wb, p_W, omega_B, v_B)
    M_add, _ = wing_inertial_terms(st)

    # independent point-mass assembly of the SAME spatial inertia about the
    # base origin, built in WORLD coords from the free-body element kinematics
    # and rotated into base coords at the end
    els = discretize(wing, 400, "equal_span")
    rho_a = props.mass / sum(e.area for e in els)
    omega_W = R_wb @ omega_B
    v_W = R_wb @ v_B
    M_W = np.zeros((6, 6))
    for el in els:
        m_el = rho_a * el.area
        st_el = element_state_free_body(wing, el, kin, 0.037, 1.5e-5,
                                        R_wb, p_W, omega_W, v_W)
        r = st_el.r_W - p_W
        cx = _skew(r)
        M_W[:3, :3] += m_el * (cx.T @ cx)
        M_W[:3, 3:] += m_el * cx
        M_W[3:, :3] += m_el * cx.T
        M_W[3:, 3:] += m_el * np.eye(3)
        M_W[:3, :3] += _element_plate_inertia(m_el, el.chord, el.area, st_el.R_W_BE)
    T = np.block([[R_wb.T, np.zeros((3, 3))], [np.zeros((3, 3)), R_wb.T]])
    # residual is element-midpoint discretization noise (O(1/n_el^2))
    np.testing.assert_allclose(M_add, T @ M_W @ T.T, rtol=1e-5, atol=1e-12)


def test_bias_matches_point_mass_assembly():
    wing = _vehicle([_wing("trapezoidal", side="right", dihedral=0.4)]).wings[0]
    kin = _kin(beta=0.3, psi_amp=0.15)
    R_wb = quat_to_matrix(quat_from_euler(0.2, -0.3, 0.1))
    p_W = np.array([0.03, -0.02, 0.05])
    omega_B = np.array([0.4, -0.7, 0.2])
    v_B = np.array([0.3, 0.1, -0.2])
    props = wing_mass_properties(wing)
    t = 0.037
    st = wing_inertial_state(wing, kin, props, t, R_wb, p_W, omega_B, v_B)
    _, Q_bias = wing_inertial_terms(st)

    els = discretize(wing, 400, "equal_span")
    rho_a = props.mass / sum(e.area for e in els)
    omega_W = R_wb @ omega_B
    v_W = R_wb @ v_B
    Q_W = np.zeros(6)
    for el in els:
        m_el = rho_a * el.area
        st_el = element_state_free_body(wing, el, kin, t, 1.5e-5,
                                        R_wb, p_W, omega_W, v_W)
        r_rel = st_el.r_W - p_W
        # st_el.a_W is the element acceleration with base acceleration zero
        F0 = -m_el * (st_el.a_W + R_wb @ np.cross(omega_B, v_B) - G_W)
        Q_W[3:] += F0
        Q_W[:3] += np.cross(r_rel, F0)
        # rotary part: per-element plate inertia; the chain angular state
        # (omega, alpha with base accel zero) is shared by all elements
        I_W = _element_plate_inertia(m_el, el.chord, el.area, st_el.R_W_BE)
        Q_W[:3] += -(I_W @ st.alpha_presc_W + np.cross(st_el.omega_W, I_W @ st_el.omega_W))
    T = np.block([[R_wb.T, np.zeros((3, 3))], [np.zeros((3, 3)), R_wb.T]])
    np.testing.assert_allclose(Q_bias, T @ Q_W, rtol=1e-5, atol=1e-12)


# --- fixed-body inertial root force ---------------------------------------------

def test_fixed_body_inertial_force_cycle_mean_is_wing_weight():
    """Periodic kinematics: mean acceleration over a cycle is zero, so the
    cycle-mean inertial root force equals the wing's weight (m*g_W)."""
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    vehicle = _vehicle([_wing("rectangular", mass=0.004)])
    aero_spec = {
        "name": "quasi_steady_blade_element",
        "coefficients": {k: _param(v_, "1") for k, v_ in {
            "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.2,
            "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
        }.items()},
    }
    vehicle = VehicleDefinition.model_validate({
        **vehicle.model_dump(), "aero": aero_spec})
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.zeros(3))
    kin = _kin(f_hz=10.0)
    ts = run_fixed_body(vehicle, kin, env, QuasiSteadyBladeElement(vehicle.aero),
                        n_wingbeats=4, steps_per_wingbeat=400)
    f_in = ts["wing_right/F_inertial"]
    # exactly one period, endpoint-exclusive (t grid lands on period multiples):
    # the discrete mean of a periodic signal over exactly one period kills all
    # non-aliased harmonics, leaving the wing weight m*g_W
    mean_f = f_in[-400:].mean(axis=0)
    np.testing.assert_allclose(mean_f, 0.004 * G_W, atol=2e-4)
    # fluctuating part is zero-mean but NOT zero (the inertial force oscillates)
    assert np.abs(f_in[-400:] - mean_f).max() > 1e-3
    # sensor channel = aero + inertial
    np.testing.assert_allclose(ts["wing_right/F_sensor"],
                               ts["wing_right/F_total"] + f_in, atol=1e-15)


# --- free-body momentum conservation ---------------------------------------------

def _free_body_plant(wings, kin):
    from fwmav_sim.aero.interface import Environment
    from fwmav_sim.simulation.free_body import AbdomenControl, FreeBodyPlant, IsolationSpec

    vehicle = _vehicle(wings)
    env = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.zeros(3))
    return FreeBodyPlant(vehicle, kin, env, None,
                         control=AbdomenControl(mode="locked"),
                         isolation=IsolationSpec())


def _body_momenta(plant, y):
    """Linear and angular momentum of the articulated bodies about the world
    origin (same construction as test_articulated_dynamics' momentum checks)."""
    from fwmav_sim.simulation.integrator import unpack_state

    nj = plant.nj
    p, quat, qj, v_W, w_B, qd_j = unpack_state(y, nj)
    quat = quat_normalize(quat)
    R_wb = quat_to_matrix(quat)
    qm = np.concatenate([p, quat, qj])
    u = np.concatenate([w_B, R_wb.T @ v_W, qd_j])
    Js = plant.model.body_jacobians(qm)
    tf = plant.model.forward_kinematics(qm)
    inertia = {b.name: np.asarray(b.inertia.value, dtype=float)
               for b in plant.vehicle.bodies}
    P = np.zeros(3)
    L = np.zeros(3)
    for name, J in zip(plant.model.body_names, Js, strict=True):
        mm = plant.model._mass[name]
        if mm == 0.0:
            continue
        V = J @ u
        R_i, o_i = tf[name]
        r_i = o_i + R_i @ plant.model._com_off[name]
        v_i = R_i @ (V[3:] + np.cross(V[:3], plant.model._com_off[name]))
        P += mm * v_i
        L += np.cross(r_i, mm * v_i) + (R_i @ inertia[name] @ R_i.T) @ (R_i @ V[:3])
    return P, L


def _wing_momenta(plant, t, y):
    """Linear and angular momentum of the massive wings about the world origin."""
    from fwmav_sim.simulation.integrator import unpack_state

    p, quat, _, v_W, w_B, _ = unpack_state(y, plant.nj)
    R_wb = quat_to_matrix(quat_normalize(quat))
    P = np.zeros(3)
    L = np.zeros(3)
    for w in plant.vehicle.wings:
        props = plant._wing_mass[w.name]
        if props is None:
            continue
        st = wing_inertial_state(w, plant.kinematics_map[w.name], props, t,
                                 R_wb, p, w_B, R_wb.T @ v_W)
        P += props.mass * st.v_c_W
        L += np.cross(st.c_W, props.mass * st.v_c_W) + st.I_cW @ st.omega_w_W
    return P, L


def _gravity_moment(plant, t, y):
    """Moment of gravity about the world origin (bodies + wings)."""
    from fwmav_sim.simulation.integrator import unpack_state

    p, quat, qj, _, _, _ = unpack_state(y, plant.nj)
    quat = quat_normalize(quat)
    R_wb = quat_to_matrix(quat)
    qm = np.concatenate([p, quat, qj])
    tf = plant.model.forward_kinematics(qm)
    mom = np.zeros(3)
    for name in plant.model.body_names:
        r_i = tf[name][1] + tf[name][0] @ plant.model._com_off[name]
        mom += np.cross(r_i, plant.model._mass[name] * G_W)
    for w in plant.vehicle.wings:
        props = plant._wing_mass[w.name]
        if props is None:
            continue
        st = wing_inertial_state(w, plant.kinematics_map[w.name], props, t,
                                 R_wb, p, np.zeros(3), np.zeros(3))
        mom += np.cross(st.c_W, props.mass * G_W)
    return mom


def test_free_body_momentum_with_wing_mass():
    from fwmav_sim.simulation.integrator import integrate_stepped, pack_state

    wings = [_wing("trapezoidal", side="right", mass=0.004, dihedral=0.3),
             _wing("trapezoidal", side="left", mass=0.004, dihedral=0.3,
                   hinge=(0.01, -0.02, -0.005))]
    kin = _kin(f_hz=8.0, beta=0.2, psi_amp=0.1)
    plant = _free_body_plant(wings, kin)
    m_tot = 0.02 + 2 * 0.004
    y0 = pack_state([0.01, -0.02, 0.03], quat_from_euler(0.1, -0.15, 0.05), [],
                    [0.05, -0.03, 0.02], [0.1, 0.2, -0.1], [])

    t, Y = integrate_stepped(plant.rhs, y0, 0.4, rtol=1e-11, atol=1e-13,
                             sample_dt=0.005)
    P0_b, L0_b = _body_momenta(plant, Y[:, 0])
    P0_w, L0_w = _wing_momenta(plant, 0.0, Y[:, 0])
    P0 = P0_b + P0_w
    L0 = L0_b + L0_w
    for k in (1, len(t) // 2, len(t) - 1):
        P_b, L_b = _body_momenta(plant, Y[:, k])
        P_w, L_w = _wing_momenta(plant, t[k], Y[:, k])
        P = P_b + P_w
        L = L_b + L_w
        # linear momentum: dP/dt = total weight (root forces are internal)
        np.testing.assert_allclose(P, P0 + m_tot * G_W * t[k], atol=1e-7)
        # angular momentum: dL/dt = moment of gravity about the world origin,
        # integrated along the trajectory (trapezoid on the recorded samples)
        m_grav = np.zeros(3)
        for j in range(k):
            m_grav += 0.5 * (_gravity_moment(plant, t[j], Y[:, j])
                             + _gravity_moment(plant, t[j + 1], Y[:, j + 1])) \
                * (t[j + 1] - t[j])
        np.testing.assert_allclose(L, L0 + m_grav, atol=2e-7)


def test_massless_wings_skip_inertia_path():
    """Massless wings: the DR-006 path is inert (M1-M5 behaviour unchanged)."""
    wings = [_wing("rectangular", side="right", mass=None),
             _wing("rectangular", side="left", mass=None, hinge=(0.01, -0.02, -0.005))]
    plant = _free_body_plant(wings, _kin())
    assert all(p is None for p in plant._wing_mass.values())
    y0 = np.zeros(plant.ny)
    y0[3] = 1.0  # unit quaternion
    dydt = plant.rhs(0.0, y0)
    assert np.all(np.isfinite(dydt))
    assert plant.last_F_wing_inertia == {}
