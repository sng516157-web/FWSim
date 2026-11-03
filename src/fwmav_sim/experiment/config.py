"""Experiment definition loading (M1: fixed-body experiments).

An experiment YAML wires a vehicle definition, kinematics, environment and
resolution together. Human-readable and versioned per brief §8.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fwmav_sim.kinematics.prescribed import PrescribedWingKinematics
from fwmav_sim.vehicle.schema import EnvironmentSpec, VehicleDefinition


class ExperimentDefinition(BaseModel):
    """Top-level experiment configuration (M1 fixed-body scope)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["fixed_body"]
    vehicle: str  # path to a VehicleDefinition YAML, relative to the experiment file
    kinematics: PrescribedWingKinematics
    environment: EnvironmentSpec
    n_wingbeats: int = Field(gt=0)
    steps_per_wingbeat: int = Field(gt=0)


def load_experiment(path: str | Path) -> ExperimentDefinition:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        return ExperimentDefinition.model_validate(yaml.safe_load(fh))


def load_vehicle(path: str | Path) -> VehicleDefinition:
    with open(path, encoding="utf-8") as fh:
        return VehicleDefinition.model_validate(yaml.safe_load(fh))
