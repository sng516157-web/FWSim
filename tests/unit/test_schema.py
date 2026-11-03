"""Unit tests for the parameter schema (M0 stage file §0.4).

The schema must reject: missing unit, unknown provenance source, negative
mass, and malformed inertia tensors (non-symmetric or non-positive-definite).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fwmav_sim.vehicle.schema import (
    AeroModelSpec,
    BodySpec,
    Matrix3Parameter,
    Parameter,
    Provenance,
    Vector3Parameter,
    VehicleDefinition,
)

_PROV = {"source": "estimated", "uncertainty_or_range": "placeholder"}
_IDENT3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _param(value: float = 1.0) -> dict:
    return {"value": value, "unit": "kg", "provenance": _PROV}


def _vec3() -> dict:
    return {"value": [0.0, 0.0, 0.0], "unit": "m", "provenance": _PROV}


def _mat3(value: list[list[float]] | None = None) -> dict:
    return {"value": value or _IDENT3, "unit": "kg m^2", "provenance": _PROV}


def _body(mass: float = 0.01, inertia: list[list[float]] | None = None) -> dict:
    return {
        "name": "thorax",
        "mass": _param(mass),
        "com_offset": _vec3(),
        "inertia": _mat3(inertia),
    }


class TestProvenance:
    def test_unknown_source_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(source="guessed")

    @pytest.mark.parametrize("source", ["measured", "literature", "estimated", "fitted"])
    def test_all_known_sources_accepted(self, source):
        assert Provenance(source=source).source == source


class TestParameter:
    def test_missing_unit_rejected(self):
        with pytest.raises(ValidationError):
            Parameter(value=1.0, provenance=_PROV)

    def test_empty_unit_rejected(self):
        with pytest.raises(ValidationError):
            Parameter(value=1.0, unit="", provenance=_PROV)

    def test_missing_provenance_rejected(self):
        with pytest.raises(ValidationError):
            Parameter(value=1.0, unit="m")


class TestBodySpec:
    def test_negative_mass_rejected(self):
        with pytest.raises(ValidationError, match="mass must be positive"):
            BodySpec(**_body(mass=-0.01))

    def test_zero_mass_rejected(self):
        with pytest.raises(ValidationError, match="mass must be positive"):
            BodySpec(**_body(mass=0.0))

    def test_non_symmetric_inertia_rejected(self):
        bad = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValidationError, match="inertia tensor must be symmetric"):
            BodySpec(**_body(inertia=bad))

    def test_non_positive_definite_inertia_rejected(self):
        bad = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValidationError, match="inertia tensor must be positive definite"):
            BodySpec(**_body(inertia=bad))

    def test_valid_body_accepted(self):
        body = BodySpec(**_body())
        assert body.mass.value == pytest.approx(0.01)


class TestVectorAndMatrixShapes:
    def test_vector3_wrong_length_rejected(self):
        with pytest.raises(ValidationError):
            Vector3Parameter(value=[1.0, 2.0], unit="m", provenance=_PROV)

    def test_matrix_not_3x3_rejected(self):
        with pytest.raises(ValidationError, match="3x3"):
            Matrix3Parameter(value=[[1.0, 0.0], [0.0, 1.0]], unit="kg m^2", provenance=_PROV)


class TestVehicleDefinition:
    def test_minimal_vehicle_accepted(self):
        vehicle = VehicleDefinition(
            name="skeleton",
            bodies=[_body()],
            aero=AeroModelSpec(name="none", coefficients={}),
        )
        assert vehicle.name == "skeleton"
        assert vehicle.blade_elements.n_elements == 16

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            VehicleDefinition(
                name="skeleton",
                bodies=[_body()],
                aero=AeroModelSpec(name="none", coefficients={}),
                bogus_field=1,
            )

    def test_json_schema_exportable(self):
        assert VehicleDefinition.model_json_schema()["title"] == "VehicleDefinition"
