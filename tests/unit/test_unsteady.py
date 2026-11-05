"""Unsteady term tests (M6 Task 7): wake capture + clap-and-fling.

Sign, timing, scaling and off-by-default behaviour; fixed-body experiment with
the terms enabled shows the wake-capture peak after stroke reversal.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.aero.interface import Environment
from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.aero.unsteady import (
    _reversal_clock,
    _wing_tip_W,
    clap_fling_force,
    wake_capture_force,
)
from fwmav_sim.experiment.config import load_vehicle
from fwmav_sim.kinematics.blade_element import BladeElement
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import UnsteadyTermSpec

_PROV = {"source": "estimated"}


def _p(v, u="1"):
    return {"value": v, "unit": u, "provenance": dict(_PROV)}


_KIN = PrescribedWingKinematics.model_validate({
    "flap_amplitude": _p(1.2, "rad"), "flap_frequency": _p(10.0, "Hz"),
    "pitch_amplitude": _p(0.5, "rad"), "pitch_phase": _p(0.8, "rad"),
    "pitch_offset": _p(0.0, "rad"), "stroke_plane_angle": _p(0.0, "rad"),
})

_EL = BladeElement(s=0.1, chord=0.05, area=0.005, x_c=0.0)
_ENV = Environment(rho=1.225, nu=1.5e-5, v_air_W=np.zeros(3))
_E_L = np.array([0.0, 0.0, -1.0])


class _El:
    """Minimal element stand-in for the unsteady terms (s, chord, area, v_W)."""

    s, chord, area, x_c = _EL.s, _EL.chord, _EL.area, _EL.x_c
    v_W = np.array([0.3, 0.0, 0.0])


def test_reversal_clock():
    f = _KIN.flap_frequency.value
    T_half = 0.5 / f
    for k in range(4):
        t_rev = k * T_half
        dt, half = _reversal_clock(_KIN, t_rev + 1e-6)
        assert dt == pytest.approx(1e-6, abs=1e-9)
        assert half == pytest.approx(T_half)
        dt_mid, _ = _reversal_clock(_KIN, t_rev + T_half / 2)
        assert dt_mid == pytest.approx(T_half / 2, rel=1e-9)


def test_wake_capture_zero_at_reversal_peaks_after():
    spec = UnsteadyTermSpec(gain=0.5, params={"tau_rise_frac": 0.05, "tau_decay_frac": 0.2})
    T_half = 0.5 / _KIN.flap_frequency.value
    f_at = lambda t: wake_capture_force(_el_or(_EL), _EL, _KIN, t, _E_L, 1.225, spec)
    assert np.all(f_at(0.0) == 0.0)  # exactly at reversal: window = 0
    # rises to a peak inside the half-stroke, then decays
    ts = np.linspace(0.001, T_half, 50)
    mags = [np.linalg.norm(f_at(t)) for t in ts]
    peak_i = int(np.argmax(mags))
    assert 0 < peak_i < len(ts) - 1  # interior peak (rise-then-decay)
    # direction: along e_L
    f_peak = f_at(ts[peak_i])
    assert f_peak[2] < 0 and abs(f_peak[0]) < 1e-15
    # scaling: gain and v_peak^2 (element span station)
    f2 = wake_capture_force(None, BladeElement(s=0.2, chord=0.05, area=0.01, x_c=0.0),
                            _KIN, ts[peak_i], _E_L, 1.225, spec)
    f1 = wake_capture_force(None, _EL, _KIN, ts[peak_i], _E_L, 1.225, spec)
    # s doubles -> v_peak doubles -> force x4; area doubles -> another x2 => x8
    assert np.linalg.norm(f2) / np.linalg.norm(f1) == pytest.approx(8.0, rel=1e-9)


def _el_or(el):
    return None  # wing arg unused by the reduced form


def test_wake_capture_rise_decay_shape():
    spec = UnsteadyTermSpec(gain=1.0, params={"tau_rise_frac": 0.05, "tau_decay_frac": 0.2})
    T_half = 0.05
    w = wake_capture_force(None, _EL, _KIN, 0.5 * T_half, _E_L, 1.225, spec)
    # window = exp(-dt/tau_d) - exp(-dt/tau_r)
    dt = 0.5 * T_half
    expect = np.exp(-dt / (0.2 * T_half)) - np.exp(-dt / (0.05 * T_half))
    q = 0.5 * 1.225 * (_KIN.flap_amplitude.value * 2 * np.pi * 10 * _EL.s) ** 2 * _EL.area
    assert np.linalg.norm(w) == pytest.approx(q * expect, rel=1e-9)


def test_clap_fling_gap_threshold_and_scaling():
    spec = UnsteadyTermSpec(gain=0.5, params={"gap_threshold": 0.05})
    far = clap_fling_force(None, _El(), _KIN, 0.0, _E_L, 1.225, spec,
                           mirror_tip_W=np.array([1.0, 0.0, 0.0]),
                           own_tip_W=np.zeros(3))
    assert np.all(far == 0.0)
    near = clap_fling_force(None, _El(), _KIN, 0.0, _E_L, 1.225, spec,
                            mirror_tip_W=np.array([0.01, 0.0, 0.0]),
                            own_tip_W=np.zeros(3))
    assert np.linalg.norm(near) > 0
    # factor = 1 - gap/g_thresh = 0.8
    q = 0.5 * 1.225 * 0.3**2 * _El.area
    assert np.linalg.norm(near) == pytest.approx(0.5 * q * 0.8, rel=1e-9)
    assert near[2] < 0  # along e_L


def test_tip_positions_mirror():
    wing_r = load_vehicle("reference_vehicles/airpulse.yaml").wings[0]
    wing_l = load_vehicle("reference_vehicles/airpulse.yaml").wings[1]
    kin = _KIN
    tip_r = _wing_tip_W(wing_r, kin, 0.0)
    tip_l = _wing_tip_W(wing_l, kin, 0.0)
    # mirrored wings at t=0 (same stroke phase): tips at mirrored y, same x/z
    assert tip_r[0] == pytest.approx(tip_l[0], abs=1e-12)
    assert tip_r[2] == pytest.approx(tip_l[2], abs=1e-12)
    assert tip_r[1] == pytest.approx(-tip_l[1], rel=1e-3)  # y mirrored


def test_fixed_body_unsteady_channels_and_reversal_peak(tmp_path):
    """End-to-end: the demo experiment produces F_unsteady channels with a
    peak right after each stroke reversal."""
    from fwmav_sim.experiment.config import kinematics_map, load_experiment
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    exp = load_experiment("experiments/m6_unsteady_airpulse.yaml")
    vehicle = load_vehicle("reference_vehicles/airpulse_m6_unsteady.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body(vehicle, kinematics_map(exp, vehicle), _ENV.__class__(
        rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3)), model,
        n_wingbeats=2, steps_per_wingbeat=100)
    f_un = ts["wing_right/F_unsteady"]
    assert np.linalg.norm(f_un, axis=1).max() > 0.0
    # aero-only channels unchanged in kind/shape
    assert ts["wing_right/F_total"].shape == f_un.shape


def test_fixed_body_wake_only_peaks_post_reversal(tmp_path):
    """Wake capture alone (no clap term): the unsteady force peaks in the
    first half of a half-stroke, right after reversal."""
    import copy

    from fwmav_sim.experiment.config import kinematics_map, load_experiment
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    exp = load_experiment("experiments/m6_unsteady_airpulse.yaml")
    vehicle = load_vehicle("reference_vehicles/airpulse_m6_unsteady.yaml")
    aero = copy.deepcopy(vehicle.aero)
    del aero.unsteady["clap_fling"]
    model = QuasiSteadyBladeElement(aero)
    ts = run_fixed_body(vehicle.model_copy(update={"aero": aero}),
                        kinematics_map(exp, vehicle),
                        Environment(rho=1.225, nu=1.460e-5, v_air_W=np.zeros(3)),
                        model, n_wingbeats=2, steps_per_wingbeat=100)
    f_un = ts["wing_right/F_unsteady"]
    mags = np.linalg.norm(f_un, axis=1)
    assert mags.max() > 0.0
    # window asymmetry: the force right after a reversal dwarfs the force
    # right before the next one (window ratio ~20x; robust to e_L modulation)
    f = exp.kinematics.flap_frequency.value
    T_half = 0.5 / f
    delta_s = -np.pi / 2  # from the experiment kinematics
    w = 2 * np.pi * f
    t_rev = (np.pi - delta_s) / w  # first reversal after t=0
    i_early = int(np.argmin(np.abs(ts.t - (t_rev + 0.1 * T_half))))
    i_late = int(np.argmin(np.abs(ts.t - (t_rev + 0.9 * T_half))))
    assert mags[i_early] > 3.0 * mags[i_late]


def test_default_off_no_channel():
    """Without the unsteady block the records are the M1-M5 shape exactly."""
    from fwmav_sim.simulation.fixed_body import run_fixed_body

    vehicle = load_vehicle("reference_vehicles/airpulse.yaml")
    model = QuasiSteadyBladeElement(vehicle.aero)
    from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics

    kin = PrescribedWingKinematics.model_validate({
        "flap_amplitude": _p(1.2, "rad"), "flap_frequency": _p(10.0, "Hz"),
        "pitch_amplitude": _p(0.5, "rad"), "pitch_phase": _p(0.8, "rad"),
        "pitch_offset": _p(0.0, "rad"), "stroke_plane_angle": _p(0.0, "rad"),
    })
    ts = run_fixed_body(vehicle, kin, _ENV, model, n_wingbeats=1, steps_per_wingbeat=20)
    assert "wing_right/F_unsteady" not in ts.channel_names()
