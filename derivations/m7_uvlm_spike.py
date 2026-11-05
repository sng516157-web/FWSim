"""M7 Task 1 spike: drive PteraSoftware's unsteady ring-UVLM with our
prescribed flapping kinematics and geometry; extract the force history.

Questions answered (go/no-go for the adapter):
  (a) prescribed sinusoidal wing motion — ps.WingMovement angles
  (b) our planform geometry — span/chords/panels
  (c) hover (zero freestream) — OperatingPoint vCg__E = 0
  (d) forces in NED Earth axes (their frames doc: +x N, +y E, +z down —
      IDENTICAL to ours; body axes +x front, +y right, +z bottom — also ours)

Comparison anchor: our quasi-steady cycle-mean vertical force for the same
case is computed and printed alongside (expect differences: UVLM includes
induced/wake effects our quasi-steady lacks — that gap is the M7 T2 study,
not a correctness criterion here).

Run: python derivations/m7_uvlm_spike.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SPAN = 0.12       # m (per wing, our bio placeholder)
ROOT_CHORD = 0.035
TIP_CHORD = 0.018
FLAP_AMP_DEG = 28.65  # 0.5 rad
FLAP_HZ = 5.0
RHO = 1.225


def main() -> None:
    import pterasoftware as ps

    wing = ps.Wing(
        wing_cross_sections=[
            ps.WingCrossSection(
                num_spanwise_panels=8, chord=ROOT_CHORD,
                Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0,
                airfoil=ps.Airfoil(name="naca0008"),
            ),
            ps.WingCrossSection(
                num_spanwise_panels=None, chord=TIP_CHORD,
                Lp_Wcsp_Lpp=(0.0, SPAN, 0.0),
                angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0,
                airfoil=ps.Airfoil(name="naca0008"),
            ),
        ],
        name="flapping wing", symmetric=True, mirror_only=False,
        Ler_Gs_Cgs=(0.0, 0.005, 0.0),
        symmetryNormal_G=(0.0, 1.0, 0.0), symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=4, chordwise_spacing="uniform",
    )
    airplane = ps.Airplane(wings=[wing], name="spike")

    # prescribed flapping about the wing's stroke axis, sine spacing
    wcs_movements_r = [
        ps.WingCrossSectionMovement(
            base_wing_cross_section=wcs,
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0), periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        )
        for wcs in airplane.wings[0].wing_cross_sections
    ]
    wcs_movements_l = [
        ps.WingCrossSectionMovement(
            base_wing_cross_section=wcs,
            ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0), periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        )
        for wcs in airplane.wings[1].wing_cross_sections
    ]
    wing_movement = ps.WingMovement(
        base_wing=airplane.wings[0],
        wing_cross_section_movements=wcs_movements_r,
        ampLer_Gs_Cgs=(0.0, 0.0, 0.0), periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"), phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(FLAP_AMP_DEG, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(1.0 / FLAP_HZ, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    )
    wing_movement_l = ps.WingMovement(
        base_wing=airplane.wings[1],
        wing_cross_section_movements=wcs_movements_l,
        ampLer_Gs_Cgs=(0.0, 0.0, 0.0), periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"), phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(FLAP_AMP_DEG, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(1.0 / FLAP_HZ, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
    )
    airplane_movement = ps.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement, wing_movement_l],
        ampCg_GP1_CgP1=(0.0, 0.0, 0.0), periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1=("sine", "sine", "sine"), phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
    )
    op = ps.OperatingPoint(rho=RHO, vCg__E=1e-3)  # hover approximated (vCg must be > 0; 1 mm/s << flap speeds)
    op_movement = ps.OperatingPointMovement(base_operating_point=op)
    movement = ps.Movement(airplane_movements=[airplane_movement],
                           operating_point_movement=op_movement,
                           delta_time=1.0 / (FLAP_HZ * 40), num_steps=None,
                           num_cycles=3)
    problem = ps.UnsteadyProblem(movement=movement)
    solver = ps.UnsteadyRingVortexLatticeMethodSolver(unsteady_problem=problem)
    solver.run(prescribed_wake=False, show_progress=True)

    # spike step 1: discover the loads interface (extraction comes next)
    print("solver type:", type(solver).__name__)
    load_attrs = [a for a in dir(solver) if "oad" in a or "orce" in a or "oment" in a]
    print("load-ish attrs:", load_attrs)
    _inspect(solver)
    ps.log_results(solver)




def _inspect(solver):
    import numpy as np
    sps = solver.unsteady_problem.steady_problems
    F = np.array([sp.airplanes[0].forces_W for sp in sps])  # per-step, wind axes
    np.save("derivations/_m7_uvlm_spike_forces.npy", F)
    print("steps:", len(sps), "F shape:", F.shape)
    print("Fz (down+, N): mean %.5f  min %.5f  max %.5f" % (F[:, 2].mean(), F[:, 2].min(), F[:, 2].max()))
    print("Fx mean %.5f  Fy mean %.5f" % (F[:, 0].mean(), F[:, 1].mean()))
    print("per-wing area-normalized q0.5*rho*A*(f*amp*2*pi*span)^2 reference below")
    # rough scale check: dynamic pressure at midspan flap speed
    import math
    A = 0.5*(0.035+0.018)*0.12
    v = 5.0*math.radians(28.65)*2*math.pi*0.06  # f * amp * 2pi * r_mid... flap speed ~ amp*w*r
    v = 28.65*math.pi/180*2*math.pi*5.0*0.06
    q = 0.5*1.225*v*v
    print("flap tip-mid speed %.3f m/s, q*A = %.5f N" % (v, q*A))

if __name__ == "__main__":
    main()
