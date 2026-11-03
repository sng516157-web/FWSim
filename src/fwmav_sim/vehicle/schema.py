"""Vehicle and simulation parameter schema (Pydantic v2).

Every physical parameter carries provenance per project brief §5.8: value, SI
unit, and a record of where the value came from. The schema enforces units and
basic physical validity (positive mass, symmetric positive-definite inertia)
at load time, not by documentation.

Symbol names used here follow docs/conventions/notation.md.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Provenance(BaseModel):
    """Origin and validity of a physical parameter (brief §5.8)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["measured", "literature", "estimated", "fitted"]
    uncertainty_or_range: str | None = None
    calibration_dataset: str | None = None
    valid_regime: str | None = None
    reference: str | None = None  # citation for literature values


class Parameter(BaseModel):
    """A scalar physical parameter with unit and provenance."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str = Field(min_length=1)  # SI; required, non-empty
    provenance: Provenance


class Vector3Parameter(BaseModel):
    """A 3-vector physical parameter (SI), components in the documented frame."""

    model_config = ConfigDict(extra="forbid")

    value: list[float] = Field(min_length=3, max_length=3)
    unit: str = Field(min_length=1)
    provenance: Provenance


class Matrix3Parameter(BaseModel):
    """A 3x3 matrix physical parameter (SI), e.g. an inertia tensor."""

    model_config = ConfigDict(extra="forbid")

    value: list[list[float]]
    unit: str = Field(min_length=1)
    provenance: Provenance

    @field_validator("value")
    @classmethod
    def _is_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("matrix parameter must be 3x3")
        return v


class BodySpec(BaseModel):
    """A rigid body: mass, CoM offset in the body frame, inertia about the CoM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mass: Parameter
    com_offset: Vector3Parameter  # r of the body CoM in the parent frame, m
    inertia: Matrix3Parameter  # about the body CoM, expressed in the body frame, kg m^2

    @field_validator("mass")
    @classmethod
    def _mass_positive(cls, m: Parameter) -> Parameter:
        if m.value <= 0:
            raise ValueError("body mass must be positive")
        return m

    @field_validator("inertia")
    @classmethod
    def _inertia_symmetric_positive_definite(cls, i: Matrix3Parameter) -> Matrix3Parameter:
        mat = np.asarray(i.value, dtype=float)
        if not np.allclose(mat, mat.T, atol=1e-15):
            raise ValueError("inertia tensor must be symmetric")
        if not np.all(np.linalg.eigvalsh(mat) > 0):
            raise ValueError("inertia tensor must be positive definite")
        return i


class JointSpec(BaseModel):
    """A joint between two bodies. MVP topology: revolute abdomen joints."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: Literal["revolute", "prismatic", "fixed"]
    parent_body: str = Field(min_length=1)
    child_body: str = Field(min_length=1)
    axis: list[float] = Field(min_length=3, max_length=3)  # unit axis in the parent frame
    location: Vector3Parameter  # joint origin in the parent frame, m
    limits: tuple[float, float] | None = None  # [rad] or [m]; lower, upper
    damping: Parameter | None = None  # viscous damping, N m s / rad

    @field_validator("axis")
    @classmethod
    def _axis_unit(cls, a: list[float]) -> list[float]:
        if not np.isclose(np.linalg.norm(a), 1.0, atol=1e-9):
            raise ValueError("joint axis must be a unit vector")
        return a

    @field_validator("limits")
    @classmethod
    def _limits_ordered(cls, lim: tuple[float, float] | None) -> tuple[float, float] | None:
        if lim is not None and lim[0] >= lim[1]:
            raise ValueError("joint limits must be (lower < upper)")
        return lim


class WingSpec(BaseModel):
    """One rigid wing (assumption A-001, A-005)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    side: Literal["right", "left"]
    span: Parameter  # single-wing span, m
    root_chord: Parameter  # m
    tip_chord: Parameter  # m
    planform: Literal["rectangular", "trapezoidal", "elliptical"]
    hinge_location: Vector3Parameter  # wing-root origin in the body frame B, m
    # chordwise location of the wing pitch (theta_w) axis: 0 = leading edge,
    # 0.5 = mid-chord (M0->M1 resolution: configurable, reference default 0.0)
    pitch_axis_fraction: Parameter = Field(
        default_factory=lambda: Parameter(
            value=0.0,
            unit="1",
            provenance=Provenance(source="estimated", reference="M1 default placeholder"),
        )
    )

    @field_validator("span", "root_chord", "tip_chord")
    @classmethod
    def _positive_length(cls, p: Parameter) -> Parameter:
        if p.value <= 0:
            raise ValueError("wing dimensions must be positive")
        return p

    @field_validator("pitch_axis_fraction")
    @classmethod
    def _pitch_axis_in_range(cls, p: Parameter) -> Parameter:
        if not 0.0 <= p.value <= 1.0:
            raise ValueError("pitch_axis_fraction must lie in [0, 1]")
        return p


class BladeElementSpec(BaseModel):
    """Spanwise discretization of each rigid wing into blade elements."""

    model_config = ConfigDict(extra="forbid")

    n_elements: int = Field(gt=0)
    discretization: Literal["equal_span", "equal_area"] = "equal_span"


class AeroModelSpec(BaseModel):
    """Aerodynamic model selection and its coefficient set with provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)  # e.g. "quasi_steady_blade_element"
    coefficients: dict[str, Parameter] = Field(default_factory=dict)


class ActuatorSpec(BaseModel):
    """Actuator driving a joint, in one of the staged modes of brief §6.4."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    joint: str = Field(min_length=1)
    mode: Literal["prescribed", "profile_limited", "torque_driven"]
    torque_limit: Parameter | None = None  # N m
    rate_limit: Parameter | None = None  # rad / s


class SolverSpec(BaseModel):
    """Numerical integration settings (brief §5.7)."""

    model_config = ConfigDict(extra="forbid")

    integrator: str = "DOP853"  # scipy solve_ivp method name
    rtol: float = Field(gt=0, default=1e-9)
    atol: float = Field(gt=0, default=1e-12)
    steps_per_wingbeat: int = Field(gt=0, default=200)
    max_time: float = Field(gt=0)  # s


class EnvironmentSpec(BaseModel):
    """Air properties and wind. Wind is uniform and steady (assumption A-006)."""

    model_config = ConfigDict(extra="forbid")

    air_density: Parameter  # kg / m^3
    kinematic_viscosity: Parameter  # nu, m^2 / s (needed for blade-element Reynolds number)
    wind: Vector3Parameter  # air velocity in the world frame W, m / s

    @field_validator("air_density", "kinematic_viscosity")
    @classmethod
    def _positive(cls, p: Parameter) -> Parameter:
        if p.value <= 0:
            raise ValueError("air density and kinematic viscosity must be positive")
        return p


class VehicleDefinition(BaseModel):
    """Complete vehicle definition: bodies, joints, wings, aero selection."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    bodies: list[BodySpec] = Field(min_length=1)
    joints: list[JointSpec] = Field(default_factory=list)
    wings: list[WingSpec] = Field(default_factory=list)
    blade_elements: BladeElementSpec = BladeElementSpec(n_elements=16)
    aero: AeroModelSpec
