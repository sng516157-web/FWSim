"""Control pipeline executor (M8 Task 0).

A pipeline is a signal-flow DAG declared in YAML:

    pipeline:
      - {id: imu, type: sensor_imu, params: {...}}
      - {id: est, type: estimator_complementary,
         inputs: [{port: imu, from: imu.measurement}]}
      - {id: ctl, type: controller_lqr, params: {...},
         inputs: [{port: state, from: est.state}]}
      - {id: mix, type: mixer_wingbeat_channels,
         inputs: [{port: channels, from: ctl.channels}]}

The executor topologically sorts the DAG and compiles it to a callable
``pipeline(state, t) -> dict`` evaluated per control step (wingbeat scale,
A-007). Every block declares typed input/output ports; a missing input, an
unknown port, a type mismatch, or a cycle fails loudly AT LOAD, never
mid-run. The same executor serves batch runs and live sessions — one data
flow, no side channels (M8 design rule 2).

Signal types (port ``signal`` tags):
    state      — vehicle state vector (full or estimated)
    imu        — IMU measurement (gyro, accel)
    mocap      — motion-capture measurement (position, attitude)
    channels   — 4-vector [thrust, roll, pitch, yaw] (ChannelCommand order)
    torque     — scalar joint torque command
    scalar     — any scalar signal (scopes, gains)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Block registry: declared ports with signal types
# ---------------------------------------------------------------------------


class Port(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal: str  # one of the signal types above
    shape: int | None = None  # vector length; None = scalar or unchecked


class BlockSpec(BaseModel):
    """Static description of a block type: its typed ports."""

    model_config = ConfigDict(frozen=True)

    inputs: dict[str, Port] = {}
    outputs: dict[str, Port] = {}


BLOCK_REGISTRY: dict[str, BlockSpec] = {
    "sensor_imu": BlockSpec(
        inputs={"state": Port(signal="state")},
        outputs={"measurement": Port(signal="imu", shape=6)},
    ),
    "sensor_mocap": BlockSpec(
        inputs={"state": Port(signal="state")},
        outputs={"measurement": Port(signal="mocap", shape=7)},
    ),
    "estimator_complementary": BlockSpec(
        inputs={"imu": Port(signal="imu", shape=6),
                "mocap": Port(signal="mocap", shape=7)},
        outputs={"state": Port(signal="state")},
    ),
    "controller_pid": BlockSpec(
        inputs={"state": Port(signal="state")},
        outputs={"channels": Port(signal="channels", shape=4)},
    ),
    "controller_lqr": BlockSpec(
        inputs={"state": Port(signal="state")},
        outputs={"channels": Port(signal="channels", shape=4)},
    ),
    "mixer_wingbeat_channels": BlockSpec(
        inputs={"channels": Port(signal="channels", shape=4)},
        outputs={"channels": Port(signal="channels", shape=4)},
    ),
    "actuator_torque": BlockSpec(
        inputs={"channels": Port(signal="channels", shape=4)},
        outputs={"torque": Port(signal="torque")},
    ),
    "scope": BlockSpec(
        inputs={"signal": Port(signal="scalar")},
        outputs={},
    ),
}

# ---------------------------------------------------------------------------
# YAML schema
# ---------------------------------------------------------------------------


class InputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: str
    from_: str = Field(alias="from")  # "block_id.port_name"


class BlockInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str
    params: dict[str, Any] = {}
    inputs: list[InputBinding] = []


class PipelineDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: list[BlockInstance]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PipelineError(ValueError):
    """Load-time pipeline validation failure (wired wrong, typed wrong,
    cyclic). Always loud, always at load."""


def _toposort(blocks: list[BlockInstance]) -> list[BlockInstance]:
    deps: dict[str, set[str]] = {}
    by_id: dict[str, BlockInstance] = {}
    for b in blocks:
        if b.id in by_id:
            raise PipelineError(f"duplicate block id {b.id!r}")
        by_id[b.id] = b
        deps[b.id] = set()
    for b in blocks:
        for inp in b.inputs:
            src = inp.from_.split(".", 1)[0]
            if src not in by_id:
                raise PipelineError(
                    f"block {b.id!r} input {inp.port!r} references unknown block {src!r}")
            deps[b.id].add(src)
    order: list[BlockInstance] = []
    ready = sorted([bid for bid, d in deps.items() if not d])
    remaining = {bid: set(d) for bid, d in deps.items()}
    while ready:
        bid = ready.pop(0)
        order.append(by_id[bid])
        for other, d in remaining.items():
            if bid in d:
                d.discard(bid)
                if not d and by_id[other] not in order and other not in ready:
                    ready.append(other)
    if len(order) != len(blocks):
        cyclic = sorted(bid for bid, d in remaining.items() if d)
        raise PipelineError(f"pipeline has a cycle involving: {', '.join(cyclic)}")
    return order


def _validate(blocks: list[BlockInstance]) -> None:
    for b in blocks:
        if b.type not in BLOCK_REGISTRY:
            raise PipelineError(
                f"block {b.id!r}: unknown type {b.type!r} "
                f"(known: {', '.join(sorted(BLOCK_REGISTRY))})")
        spec = BLOCK_REGISTRY[b.type]
        wired = {inp.port for inp in b.inputs}
        for req, port in spec.inputs.items():
            # state-signal ports are fed by the plant state (executor-injected)
            # unless explicitly wired — they are never "required" wiring.
            if port.signal == "state":
                continue
            if req not in wired:
                raise PipelineError(
                    f"block {b.id!r} ({b.type}): required input port {req!r} is not wired")
        for inp in b.inputs:
            if inp.port not in spec.inputs:
                raise PipelineError(
                    f"block {b.id!r} ({b.type}): unknown input port {inp.port!r}")
            if "." not in inp.from_:
                raise PipelineError(
                    f"block {b.id!r} input {inp.port!r}: 'from' must be block.port, "
                    f"got {inp.from_!r}")
            src_id, src_port = inp.from_.split(".", 1)
            # source existence checked in toposort; here: port + type
            src_block = next((x for x in blocks if x.id == src_id), None)
            if src_block is None:
                continue  # toposort raises the loud error
            src_spec = BLOCK_REGISTRY.get(src_block.type)
            if src_spec is None or src_port not in src_spec.outputs:
                raise PipelineError(
                    f"block {b.id!r} input {inp.port!r}: {src_id!r} has no output "
                    f"port {src_port!r}")
            want, got = spec.inputs[inp.port], src_spec.outputs[src_port]
            if want.signal != got.signal:
                raise PipelineError(
                    f"type mismatch: {b.id}.{inp.port} expects signal {want.signal!r} "
                    f"but {src_id}.{src_port} produces {got.signal!r}")
            if want.shape is not None and got.shape is not None \
                    and want.shape != got.shape:
                raise PipelineError(
                    f"shape mismatch: {b.id}.{inp.port} expects length {want.shape} "
                    f"but {src_id}.{src_port} produces length {got.shape}")


# ---------------------------------------------------------------------------
# Block factories: params + inputs -> output dict. Each factory returns
# step(inputs: dict[str, np.ndarray], state: np.ndarray, t: float) -> dict.
# ---------------------------------------------------------------------------


def _make_block(inst: BlockInstance) -> Callable:
    from fwmav_sim.control import blocks as _blocks

    factory = getattr(_blocks, f"make_{inst.type}", None)
    if factory is None:
        raise PipelineError(f"block type {inst.type!r} is registered but has no "
                            f"factory (control.blocks.make_{inst.type} missing)")
    return factory(inst.params)


class Pipeline:
    """A compiled pipeline: topo-sorted blocks evaluated in order.

    ``pipeline(state, t)`` returns a dict of every block's outputs plus a
    ``commands`` entry (the outputs of the terminal block(s) — blocks whose
    outputs feed nothing)."""

    def __init__(self, blocks: list[BlockInstance]):
        _validate(blocks)
        self.order = _toposort(blocks)
        self._steps = {b.id: _make_block(b) for b in self.order}
        self._bindings = {b.id: {i.port: i.from_ for i in b.inputs} for b in self.order}
        self._types = {b.id: b.type for b in self.order}
        consumers = {i.from_ for b in self.order for i in b.inputs}
        self._terminals = [
            b.id for b in self.order
            if not any(f"{b.id}.{port}" in consumers
                       for port in BLOCK_REGISTRY[b.type].outputs)]

    @classmethod
    def from_yaml(cls, path) -> "Pipeline":
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(PipelineDefinition.model_validate(raw).pipeline)

    def __call__(self, state: np.ndarray, t: float) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for b in self.order:
            inputs = {}
            for port, ref in self._bindings[b.id].items():
                src_id, src_port = ref.split(".", 1)
                inputs[port] = values[f"{src_id}.{src_port}"]
            # unwired state-signal ports receive the plant state
            for port, pspec in BLOCK_REGISTRY[b.type].inputs.items():
                if pspec.signal == "state" and port not in inputs:
                    inputs[port] = state
            out = self._steps[b.id](inputs, state, t)
            for k, v in out.items():
                values[f"{b.id}.{k}"] = v
        values["commands"] = {
            tid: {port: values[f"{tid}.{port}"]
                  for port in BLOCK_REGISTRY[self._types[tid]].outputs}
            for tid in self._terminals}
        return values
