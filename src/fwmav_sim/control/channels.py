"""Wing-beat-modulation control channels (M7 Task 4; Doman et al. 2010, JGCD
33:724 — the wingbeat-modulation convention flappy uses).

Controllers should see a mechanism-independent interface: four channels
[thrust, roll, pitch, yaw] that map onto per-wing kinematics modulations:

    thrust : scale flap amplitude of all wings        (more force)
    roll   : differential flap amplitude (L vs R)     (roll torque)
    pitch  : pitch_offset shift on all wings          (fore/aft force tilt)
    yaw    : split-cycle stroke-rate asymmetry        (yaw torque)

The mixer translates channel commands into per-wing kinematics adjustments
applied at each wingbeat boundary (commands are per-beat, not per-instant —
control at wingbeat scale, per the A-007 honesty chain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelCommand:
    """One wingbeat-scale control command (all dimensionless fractions)."""

    thrust: float = 0.0  # flap amplitude scale delta (1 + thrust)
    roll: float = 0.0  # differential amplitude fraction (left -= , right +=)
    pitch: float = 0.0  # pitch_offset shift [rad] on all wings
    yaw: float = 0.0  # split-cycle asymmetry fraction


class ChannelMixer:
    """Maps ChannelCommand -> per-wing kinematics deltas (documented above)."""

    def __init__(self, wing_names: list[str]):
        self.wing_names = list(wing_names)

    def apply(self, kinematics: dict, cmd: ChannelCommand) -> dict:
        """Return a NEW per-wing kinematics dict with the command applied
        (inputs untouched). Each wing gets its own deep copy — broadcast
        kinematics share one object per side, and asymmetric commands
        (roll/yaw) would otherwise corrupt the shared object."""
        import copy

        out = {name: copy.deepcopy(kin) for name, kin in kinematics.items()}
        for name in self.wing_names:
            kin = out[name]
            side_sign = 1.0 if "right" in name else -1.0
            kin.flap_amplitude.value *= (1.0 + cmd.thrust) * (
                1.0 + side_sign * cmd.roll)
            kin.pitch_offset.value += cmd.pitch
            # split-cycle: rate asymmetry via stroke_phase offset (documented
            # approximation: a true split cycle needs a non-sinusoidal stroke;
            # phase offset is the reduced approximation)
            if kin.stroke_phase is not None:
                kin.stroke_phase.value += side_sign * cmd.yaw
            else:
                from fwmav_sim.vehicle.schema import Parameter, Provenance

                kin.stroke_phase = Parameter(
                    value=side_sign * cmd.yaw, unit="rad",
                    provenance=Provenance(source="estimated",
                                          reference="channel mixer split-cycle approximation"))
        return out
