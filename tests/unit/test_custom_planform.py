"""Custom planform (station-defined outline) tests (M9 T0)."""

from __future__ import annotations

import numpy as np
import pytest

from fwmav_sim.dynamics.wing_inertia import wing_mass_properties
from fwmav_sim.kinematics.blade_element import (
    chord_at,
    cumulative_area,
    discretize,
    le_offset_at,
    planform_area,
)
from fwmav_sim.vehicle.schema import VehicleDefinition


def _vehicle_with_wing(wing: dict) -> VehicleDefinition:
    prov = {"source": "estimated"}
    p = lambda v, u: {"value": v, "unit": u, "provenance": prov}  # noqa: E731
    return VehicleDefinition.model_validate({
        "name": "custom_planform_test",
        "data_status": "estimated",
        "bodies": [{"name": "body", "mass": p(0.02, "kg"),
                    "com_offset": p([0, 0, 0], "m"),
                    "inertia": p([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]], "kg m^2")}],
        "joints": [],
        "wings": [wing],
        "blade_elements": {"n_elements": 16, "discretization": "equal_span"},
        "aero": {"name": "none", "coefficients": {}},
    })


def _custom_wing(stations, planform="custom", **kw):
    prov = {"source": "estimated"}
    p = lambda v, u: {"value": v, "unit": u, "provenance": prov}  # noqa: E731
    wing = {
        "name": "wing_right", "side": "right",
        "span": p(stations[-1]["y"], "m"),
        "root_chord": p(stations[0]["chord"], "m"),
        "tip_chord": p(stations[-1]["chord"], "m"),
        "planform": planform,
        "hinge_location": p([0, 0, 0], "m"),
        "pitch_axis_fraction": p(0.25, "1"),
    }
    if planform == "custom":
        wing["planform_stations"] = stations
    wing.update(kw)
    return wing


RECT_STATIONS = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                 {"y": 0.3, "chord": 0.2, "le_offset": 0.05}]


def test_custom_matches_rectangular():
    """A rectangular outline as stations must reproduce the analytic numbers."""
    v_custom = _vehicle_with_wing(_custom_wing(RECT_STATIONS))
    w_c = v_custom.wings[0]
    v_rect = _vehicle_with_wing(_custom_wing(RECT_STATIONS, planform="rectangular"))
    w_r = v_rect.wings[0]
    for s in [0.0, 0.1, 0.15, 0.29, 0.3]:
        assert chord_at(w_c, s) == pytest.approx(chord_at(w_r, s))
        assert cumulative_area(w_c, s) == pytest.approx(cumulative_area(w_r, s))
    assert planform_area(w_c) == pytest.approx(planform_area(w_r))
    # le_offset carries f_pa * chord for unswept analytic wings
    assert le_offset_at(w_r, 0.1) == pytest.approx(0.25 * 0.2)


def test_custom_trapezoid_area_and_chord():
    st = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
          {"y": 0.3, "chord": 0.1, "le_offset": 0.08}]
    w = _vehicle_with_wing(_custom_wing(st)).wings[0]
    assert chord_at(w, 0.15) == pytest.approx(0.15)
    assert planform_area(w) == pytest.approx(0.5 * (0.2 + 0.1) * 0.3)
    assert le_offset_at(w, 0.15) == pytest.approx(0.065)


def test_custom_discretize_area_conserving():
    st = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
          {"y": 0.15, "chord": 0.18, "le_offset": 0.04},
          {"y": 0.3, "chord": 0.05, "le_offset": -0.07}]  # swept BACK
    w = _vehicle_with_wing(_custom_wing(st)).wings[0]
    els = discretize(w, 16, "equal_span")
    assert sum(e.area for e in els) == pytest.approx(planform_area(w))
    # sweep-back moves the outer element centres to smaller x (away from LE)
    assert els[-1].x_c < els[0].x_c


def test_custom_equal_area_discretize():
    st = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
          {"y": 0.3, "chord": 0.1, "le_offset": 0.08}]
    w = _vehicle_with_wing(_custom_wing(st)).wings[0]
    els = discretize(w, 8, "equal_area")
    areas = [e.area for e in els]
    assert max(areas) == pytest.approx(min(areas), rel=1e-6)


def test_custom_wing_inertia_vs_rectangular():
    """Unswept custom outline with le_offset = f_pa * c must match the
    analytic rectangular inertia (the quadrature path vs the closed form)."""
    m = {"value": 0.004, "unit": "kg", "provenance": {"source": "estimated"}}
    w_c = _vehicle_with_wing(_custom_wing(RECT_STATIONS, mass=m)).wings[0]
    w_r = _vehicle_with_wing(_custom_wing(RECT_STATIONS, planform="rectangular",
                                          mass=m)).wings[0]
    pc, pr = wing_mass_properties(w_c), wing_mass_properties(w_r)
    np.testing.assert_allclose(pc.r_com_WR, pr.r_com_WR, atol=1e-12)
    np.testing.assert_allclose(pc.I_com_WR, pr.I_com_WR, atol=1e-14)


def test_swept_wing_com_moves_back():
    st_fwd = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
              {"y": 0.3, "chord": 0.2, "le_offset": 0.05}]
    st_swept = [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                {"y": 0.3, "chord": 0.2, "le_offset": 0.15}]
    m = {"value": 0.004, "unit": "kg", "provenance": {"source": "estimated"}}
    w0 = _vehicle_with_wing(_custom_wing(st_fwd, mass=m)).wings[0]
    w1 = _vehicle_with_wing(_custom_wing(st_swept, mass=m)).wings[0]
    assert wing_mass_properties(w1).r_com_WR[0] > wing_mass_properties(w0).r_com_WR[0]


def test_custom_validation_errors():
    # no stations at all — build the wing dict directly (helper needs stations)
    prov = {"source": "estimated"}
    p = lambda v, u: {"value": v, "unit": u, "provenance": prov}  # noqa: E731
    bare = {"name": "wing_right", "side": "right",
            "span": p(0.3, "m"), "root_chord": p(0.2, "m"), "tip_chord": p(0.2, "m"),
            "planform": "custom", "hinge_location": p([0, 0, 0], "m"),
            "pitch_axis_fraction": p(0.25, "1")}
    with pytest.raises(ValueError, match="planform_stations"):
        _vehicle_with_wing(bare)
    # stations not reaching the span (helper masks this — build by hand)
    bad_span = {"name": "wing_right", "side": "right",
                "span": p(0.3, "m"), "root_chord": p(0.2, "m"), "tip_chord": p(0.2, "m"),
                "planform": "custom", "hinge_location": p([0, 0, 0], "m"),
                "pitch_axis_fraction": p(0.25, "1"),
                "planform_stations": [{"y": 0.0, "chord": 0.2, "le_offset": 0.05},
                                      {"y": 0.1, "chord": 0.2, "le_offset": 0.05}]}
    with pytest.raises(ValueError, match="span"):
        _vehicle_with_wing(bad_span)
    # stations without custom planform
    w = _custom_wing(RECT_STATIONS, planform="rectangular")
    w["planform_stations"] = RECT_STATIONS
    with pytest.raises(ValueError, match="custom"):
        _vehicle_with_wing(w)
