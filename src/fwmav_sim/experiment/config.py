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

    mode: Literal["free", "locked", "prescribed", "profile_limited", "torque_driven"] = "locked"
    profile: AbdomenProfileSpec | None = None
    # stage 2 (profile_limited): cascaded rate/acceleration limiter
    rate_max: float | None = None  # rad / s
    accel_max: float | None = None  # rad / s^2
    # stage 3 (torque_driven): saturated torque source, sinusoid command
    tau_max: float | None = None  # N m
    tau_amplitude: float = 0.0  # N m
    tau_frequency: float = 0.0  # Hz
    joint_damping: float = 0.0  # N m s / rad


class IsolationFlagsSpec(BaseModel):
    """Mechanism-isolation switches (brief §6.6). abdomen_aero_enabled is a
    Level-3 stub: enabling it raises NotImplementedError."""

    model_config = ConfigDict(extra="forbid")

    mass_shift_enabled: bool = True
    inertial_coupling_enabled: bool = True
    abdomen_aero_enabled: bool = False
    wing_inertia_enabled: bool = True  # no-op when all wings are massless


class InitialStateFreeSpec(BaseModel):
    """Structured initial state for free_body runs."""

    model_config = ConfigDict(extra="forbid")

    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    euler: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # psi, theta, phi_r
    joint_angles: list[float] = Field(default_factory=list)
    velocity_W: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    omega_B: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_rates: list[float] = Field(default_factory=list)


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


class ClosedLoopSpec(BaseModel):
    """Closed-loop hover demo on the reduced model (M6 Task 8)."""

    model_config = ConfigDict(extra="forbid")

    forces_record: str  # cycle-averaged forces JSON (M1 record), relative path
    q_diag: list[float] = Field(min_length=8, max_length=8)  # LQR state weights
    r_diag: list[float] = Field(min_length=2, max_length=2)  # LQR control weights
    initial_perturbation: list[float] = Field(default_factory=lambda: [0.0] * 8)


class BaseConstraintSpec(BaseModel):
    """Base constraint as experiment data (DR-007): the rig/free/pivot
    choice is an option on every experiment, not a kind.

    mode: fixed (clamped rig), free (6-DoF flight), pivot (suspended).
    release_time (only with mode "free"): clamp the base DoFs until this
    time, then release — the launch transient.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed", "free", "pivot"]
    pivot_body_point: list[float] | None = Field(default=None, min_length=2, max_length=2)
    release_time: float | None = Field(default=None, gt=0.0)


class ExperimentDefinition(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["fixed_body", "suspended_pendulum", "free_reduced", "free_body",
                  "closed_loop_reduced"]
    vehicle: str | None = None  # path relative to the experiment file

    # fixed_body / free_body (aero runs)
    kinematics: PrescribedWingKinematics | dict[str, PrescribedWingKinematics] | None = None
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
    initial_state_free: InitialStateFreeSpec | None = None
    isolation: IsolationFlagsSpec | None = None
    closed_loop: ClosedLoopSpec | None = None
    base: BaseConstraintSpec | None = None  # DR-007: constraint as data
    solver: SolverSpec | None = None


def effective_base(exp: ExperimentDefinition) -> tuple[str, list[str]]:
    """Resolve the effective base constraint (DR-007).

    Returns (mode, warnings). Precedence: an explicit `base` block overrides
    the kind's implied constraint (recorded as a warning); otherwise the
    kind implies: fixed_body -> fixed, free_body/closed_loop_reduced -> free,
    suspended_pendulum -> pivot, free_reduced -> free (reduced plant).
    """
    implied = {"fixed_body": "fixed", "free_body": "free",
               "suspended_pendulum": "pivot", "free_reduced": "free",
               "closed_loop_reduced": "free"}[exp.kind]
    warnings: list[str] = []
    if exp.base is None:
        return implied, warnings
    mode = exp.base.mode
    if mode != implied:
        warnings.append(
            f"base block overrides the {exp.kind} kind's implied constraint "
            f"({implied} -> {mode}); recorded, not silent (DR-007)")
    if exp.base.release_time is not None and mode != "free":
        raise ValueError("base.release_time is only meaningful with mode: free")
    if mode == "pivot" and exp.base.pivot_body_point is None \
            and exp.suspension is None:
        raise ValueError("base mode pivot requires pivot_body_point or a suspension block")
    return mode, warnings


def kinematics_map(exp: ExperimentDefinition, vehicle: VehicleDefinition):
    """Resolve experiment kinematics to {wing_name: PrescribedWingKinematics}.

    A single kinematics block broadcasts to all wings; a mapping gives each
    wing its own (e.g. X-wing fore/aft anti-phase pairs). All wings must share
    the flapping frequency (one global wingbeat period; phases may differ).
    """
    kin = exp.kinematics
    if kin is None:
        raise ValueError("experiment requires kinematics")
    if isinstance(kin, dict):
        missing = [w.name for w in vehicle.wings if w.name not in kin]
        if missing:
            raise ValueError(f"kinematics mapping missing wings: {missing}")
        freqs = {round(kin[w.name].flap_frequency.value, 12) for w in vehicle.wings}
        if len(freqs) != 1:
            raise ValueError("all wings must share the flapping frequency")
        return dict(kin)
    return {w.name: kin for w in vehicle.wings}


def load_experiment(path: str | Path) -> ExperimentDefinition:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    from fwmav_sim.vehicle.compact import expand_experiment, is_compact

    if is_compact(raw):
        raw = expand_experiment(raw)
    return ExperimentDefinition.model_validate(raw)


def load_vehicle(path: str | Path) -> VehicleDefinition:
    from fwmav_sim.vehicle.compact import expand_compact, is_compact

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if is_compact(raw):
        raw = expand_compact(raw)
    return VehicleDefinition.model_validate(raw)
