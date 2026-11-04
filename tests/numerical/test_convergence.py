"""§11.3 convergence tests (M1 Task 12): timestep and blade-element refinement.

Real tests on a small sweep (fast); the full 5x6 grid for the report is
experiments/m1_convergence.yaml. `test_solver_tolerance_sensitivity` remains
an M3 stub (no ODE solve in M1).
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
from fwmav_sim.analysis.averaging import wingbeat_mean
from fwmav_sim.environment import still_air
from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.simulation.fixed_body import run_fixed_body
from fwmav_sim.vehicle.schema import (
    AeroModelSpec,
    BladeElementSpec,
    VehicleDefinition,
    WingSpec,
)

_PROV_EST = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}
_PROV_LIT = {"source": "literature", "reference": "Dickinson et al. 1999, Science 284:1954"}
_COEFFS = {
    "cl_0": 0.225, "cl_slope": 1.58, "cl_freq": 2.13, "cl_phase_deg": 7.20,
    "cd_0": 1.92, "cd_slope": 1.55, "cd_freq": 2.04, "cd_phase_deg": 9.82,
}


def _param(value: float, unit: str = "m") -> dict:
    return {"value": value, "unit": unit, "provenance": _PROV_EST}


def _vehicle(n_elements: int) -> VehicleDefinition:
    wing = WingSpec(
        name="wing_right", side="right",
        span=_param(0.3), root_chord=_param(0.08), tip_chord=_param(0.04),
        planform="trapezoidal",
        hinge_location={"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV_EST},
        pitch_axis_fraction=_param(0.25, "1"),
    )
    return VehicleDefinition(
        name="convergence_test",
        bodies=[{
            "name": "thorax", "mass": _param(0.01, "kg"),
            "com_offset": {"value": [0, 0, 0], "unit": "m", "provenance": _PROV_EST},
            "inertia": {"value": [[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]],
                        "unit": "kg m^2", "provenance": _PROV_EST},
        }],
        wings=[wing],
        blade_elements=BladeElementSpec(n_elements=n_elements),
        aero=AeroModelSpec(
            name="quasi_steady_blade_element",
            coefficients={k: {"value": v, "unit": "1", "provenance": _PROV_LIT}
                          for k, v in _COEFFS.items()},
        ),
    )


def _kin() -> PrescribedWingKinematics:
    return PrescribedWingKinematics(
        flap_amplitude=_param(1.0, "rad"), flap_frequency=_param(12.0, "Hz"),
        pitch_amplitude=_param(0.5, "rad"), pitch_phase=_param(np.pi, "rad"),
        pitch_offset=_param(0.1, "rad"), stroke_plane_angle=_param(0.15, "rad"),
    )


def _mean_force(n_elements: int, steps: int) -> np.ndarray:
    vehicle, kin, env = _vehicle(n_elements), _kin(), still_air()
    model = QuasiSteadyBladeElement(vehicle.aero)
    ts = run_fixed_body(vehicle, kin, env, model, n_wingbeats=2, steps_per_wingbeat=steps)
    mean, _ = wingbeat_mean(ts, "wing_right/F_total", ts.meta["wingbeat_period_s"])
    return np.asarray(mean)


def test_timestep_refinement():
    """Halving dt repeatedly must shrink the change in cycle-averaged force.

    Threshold note: the rectangle-rule cycle mean is limited by the |v_rel|
    cusp at stroke reversal, giving algebraic (not spectral) convergence;
    the observed 400-vs-800 step relative change is ~2e-4. The bound below is
    set at 5e-4 and the full trend is reported in the M1 convergence table.
    """
    ref = _mean_force(16, 800)
    e_coarse = np.linalg.norm(_mean_force(16, 100) - ref)
    e_mid = np.linalg.norm(_mean_force(16, 200) - ref)
    e_fine = np.linalg.norm(_mean_force(16, 400) - ref)
    assert e_mid < e_coarse and e_fine < e_mid
    assert e_fine / np.linalg.norm(ref) < 5e-4


def test_blade_element_refinement():
    """Refining the spanwise discretization must converge the averaged force."""
    ref = _mean_force(64, 200)
    e_coarse = np.linalg.norm(_mean_force(4, 200) - ref)
    e_mid = np.linalg.norm(_mean_force(8, 200) - ref)
    e_fine = np.linalg.norm(_mean_force(16, 200) - ref)
    assert e_mid < e_coarse and e_fine < e_mid


def test_solver_tolerance_sensitivity():
    """Free-body hover at rtol 1e-6 / 1e-9 / 1e-12: the terminal state metric
    must converge as the tolerance tightens (M3 numerical inventory)."""
    from fwmav_sim.aero.quasi_steady import QuasiSteadyBladeElement
    from fwmav_sim.core.transforms import quat_from_euler
    from fwmav_sim.environment.spec import environment_from_spec
    from fwmav_sim.experiment.config import load_experiment, load_vehicle
    from fwmav_sim.simulation.free_body import AbdomenControl, FreeBodyPlant
    from fwmav_sim.simulation.integrator import pack_state
    from fwmav_sim.simulation.runner import run_free_body
    from fwmav_sim.vehicle.schema import SolverSpec

    exp = load_experiment("experiments/m3_free_body_hover.yaml")
    vehicle = load_vehicle("reference_vehicles/bio_abdomen_fwmav.yaml")
    env = environment_from_spec(exp.environment)
    model = QuasiSteadyBladeElement(vehicle.aero)
    plant = FreeBodyPlant(vehicle, exp.kinematics, env, model,
                          control=AbdomenControl(mode="locked"))
    y0 = np.concatenate([pack_state([0, 0, 0], quat_from_euler(0, 0, 0), [0, 0],
                                    [0, 0, 0], [0, 0, 0], [0, 0])])

    def metric(rtol):
        spec = SolverSpec(integrator="DOP853", rtol=rtol, atol=rtol * 1e-3,
                          steps_per_wingbeat=200, max_time=0.05)
        ts, status, _ = run_free_body(plant, spec, y0)
        assert status == "OK"
        return np.concatenate([ts["p"][-1], ts["omega_B"][-1]])

    ref = metric(1e-12)
    e_coarse = np.linalg.norm(metric(1e-6) - ref) / np.linalg.norm(ref)
    e_fine = np.linalg.norm(metric(1e-9) - ref) / np.linalg.norm(ref)
    assert e_fine < e_coarse
    assert e_fine < 1e-3