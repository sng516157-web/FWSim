"""Experiment definition loading.

An experiment YAML wires a vehicle definition, kinematics, environment and
resolution together. Human-readable and versioned per brief §8.2.

Kinds:
  fixed_body          — M1: prescribed wing kinematics, no body motion
  suspended_pendulum  — M2: rigid-pivot suspension constraint model
                        (constraints/suspended.py); NOT free flight
  free_reduced        — M2: free reduced longitudinal model (8-state),
                        incl. the RL-sandbox foil configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import EnvironmentSpec, SolverSpec, VehicleDefinition


class SuspensionSpec(BaseModel):
    """Rigid-pivot suspension: pivot at a body-frame point (x, z) in B."""

    model_config = ConfigDict(extra="forbid")

    pivot_body_point: list[float] = Field(min_length=2, max_length=2)  # (x, z) in B, m


class AbdomenProfileSpec(BaseModel):
    """Serialized AbdomenProfile (actuation/prescribed_abdomen.py)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["step", "ramp", "sinusoid"]
    amplitude: float = 0.0
    offset: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0
    t_start: float = 0.0
    t_end: float = 1.0


class AbdomenSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["free", "prescribed", "locked"] = "locked"
    profile: AbdomenProfileSpec | None = None


class DampingSpec(BaseModel):
    """Reduced-model damping slots (M2 stage §2.4a requirement)."""

    model_config = ConfigDict(extra="forbid")

    c_lin: float = 0.0  # N s / m
    c_ang: float = 0.0  # N m s / rad
    c_a: float = 0.0  # N m s / rad


class ForceInjectionSpec(BaseModel):
    """Cycle-averaged force injection into the reduced model (Task 7).

    source:
      none                  — gravity only
      m1_record             — load a saved average-forces JSON
                              (dynamics/cycle_averaged.py), applied at the
                              wing-root hinges
      synthetic_thrust_law  — foil only: T = k_thrust * f per thruster along
                              -z_B at x = +/-arm (labelled synthetic, source:
                              estimated; NOT physics)
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["none", "m1_record", "synthetic_thrust_law"] = "none"
    record_path: str | None = None  # average-forces JSON, relative to experiment file
    k_thrust: float | None = None  # N / Hz (synthetic)
    base_f_hz: float | None = None
    freq_delta_hz: float = 0.0
    arm: float | None = None  # m


class ReducedParamsSpec(BaseModel):
    """Explicit reduced-model parameters (used when no vehicle file applies,
    e.g. the foil). All values carry provenance at the YAML level."""

    model_config = ConfigDict(extra="forbid")

    m_t: float = Field(gt=0)
    m_a: float = Field(ge=0.0)
    I_t: float = Field(gt=0)
    d: float = 0.0
    l_a: float = 0.0
    lock_abdomen: bool = False


class ExperimentDefinition(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["fixed_body", "suspended_pendulum", "free_reduced"]
    vehicle: str | None = None  # path relative to the experiment file

    # fixed_body
    kinematics: PrescribedWingKinematics | None = None
    environment: EnvironmentSpec | None = None
    n_wingbeats: int | None = Field(default=None, gt=0)
    steps_per_wingbeat: int | None = Field(default=None, gt=0)

    # dynamic kinds
    suspension: SuspensionSpec | None = None
    abdomen: AbdomenSpec | None = None
    forces: ForceInjectionSpec | None = None
    reduced_params: ReducedParamsSpec | None = None
    damping: DampingSpec | None = None
    initial_state: list[float] | None = None
    solver: SolverSpec | None = None


def load_experiment(path: str | Path) -> ExperimentDefinition:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        return ExperimentDefinition.model_validate(yaml.safe_load(fh))


def load_vehicle(path: str | Path) -> VehicleDefinition:
    with open(path, encoding="utf-8") as fh:
        return VehicleDefinition.model_validate(yaml.safe_load(fh))
