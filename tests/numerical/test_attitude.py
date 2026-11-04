"""§11.3 attitude normalization test (M3 Task 3) — real implementation."""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import quat_from_euler
from fwmav_sim.dynamics.spatial import ArticulatedModel
from fwmav_sim.simulation.integrator import GuardSpec, integrate_stepped, make_rhs, pack_state
from fwmav_sim.vehicle.schema import VehicleDefinition

_PROV = {"source": "estimated", "uncertainty_or_range": "synthetic test value"}


def _tumbling_vehicle() -> VehicleDefinition:
    return VehicleDefinition.model_validate({
        "name": "tumble_test",
        "bodies": [
            {"name": "thorax",
             "mass": {"value": 0.011, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[2e-6, 0, 0], [0, 3e-6, 0], [0, 0, 1e-6]],
                         "unit": "kg m^2", "provenance": _PROV}},
            {"name": "abdomen_roll_carrier",
             "mass": {"value": 0.0, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[0.0] * 3] * 3, "unit": "kg m^2", "provenance": _PROV},
             "massless": True},
            {"name": "abdomen",
             "mass": {"value": 0.004, "unit": "kg", "provenance": _PROV},
             "com_offset": {"value": [-0.015, 0.0, 0.0], "unit": "m", "provenance": _PROV},
             "inertia": {"value": [[5e-7, 0, 0], [0, 5e-7, 0], [0, 0, 5e-7]],
                         "unit": "kg m^2", "provenance": _PROV}},
        ],
        "joints": [
            {"name": "abdomen_pitch", "type": "revolute", "parent_body": "thorax",
             "child_body": "abdomen_roll_carrier", "axis": [0.0, 1.0, 0.0],
             "location": {"value": [-0.015, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
            {"name": "abdomen_roll", "type": "revolute", "parent_body": "abdomen_roll_carrier",
             "child_body": "abdomen", "axis": [1.0, 0.0, 0.0],
             "location": {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV}},
        ],
        "wings": [], "aero": {"name": "none", "coefficients": {}},
    })


def test_quaternion_norm_post_step():
    """Fast tumbling free body: |1 - ||quat|| < 1e-12 at every accepted step
    (per-step renormalization active, brief §5.7)."""
    model = ArticulatedModel(_tumbling_vehicle())
    y0 = pack_state([0.0, 0.0, 0.0], quat_from_euler(0.3, 0.2, 0.1), [0.2, -0.1],
                    [0.0, 0.0, 0.0], [50.0, -40.0, 30.0], [0.0, 0.0])  # fast tumble
    rhs = make_rhs(model, gravity=False)
    # dense output off: samples ARE the (renormalized) accepted steps
    t, Y = integrate_stepped(rhs, y0, 0.5, rtol=1e-9, atol=1e-12, max_step=1e-3)
    norms = np.linalg.norm(Y[3:7], axis=0)
    assert np.all(np.abs(1.0 - norms) < 1e-12)
    # and the run actually tumbled (not a trivial no-op)
    assert np.abs(Y[10:13]).max() > 20.0


def test_quat_guard_triggers():
    """A deliberately lax guard must fire when renormalization is off-scale."""
    model = ArticulatedModel(_tumbling_vehicle())
    y0 = pack_state([0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0],
                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0])
    rhs = make_rhs(model, gravity=False)
    from fwmav_sim.simulation.integrator import SolverFailure

    try:
        integrate_stepped(rhs, y0, 0.01, guard=GuardSpec(max_quat_norm_dev=1e-12))
    except SolverFailure:
        return
    raise AssertionError("guard should fire on a non-unit quaternion state")
