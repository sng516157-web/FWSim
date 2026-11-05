"""PteraSoftware UVLM backend (M7 Task 1).

Wraps PteraSoftware's unsteady ring-UVLM as an optional aerodynamic tier
above the quasi-steady blade-element model. UVLM solves the WHOLE time
window at once (the wake needs motion history), so this is a batch solver
for prescribed-kinematics runs — it cannot answer element_forces()
pointwise. Scope (recorded): fixed_body rig runs only; free-body lockstep
coupling is a follow-up (their own free-flight path is MuJoCo-bound).

Geometry mapping (the non-obvious part, learned in the spike):
- ptera forbids negative span: a left wing cannot extend -y. Each of our
  right wings is therefore paired with its identical left mirror into ONE
  symmetric ptera Wing with the root offset from the symmetry plane — their
  type-5 symmetry machinery then splits it into the mirrored pair. Our
  per-side hinge y-offsets map onto that root offset exactly.
- Dihedral (DR-005): the base (right) wing's axes rotate by -gamma about x
  (our sigma=-1 right convention); the reflection carries the left side.
- Asymmetric pairs (different geometry per side) are rejected — recorded
  limitation.

Frame mapping (their docs/AXES_POINTS_AND_FRAMES.md == our frames.md):
Earth axes both NED (+x N, +y E, +z down); body axes both +x front, +y right,
+z bottom. Wing kinematics: our stroke phi maps to their x-rotation amplitude
(flap about the forward axis), our pitch theta_w to their y-rotation, our
deviation psi_w to their z-rotation. Our cos-based stroke maps onto their
sine spacing with a +90 deg phase shift.

Dependency note: pterasoftware is an OPTIONAL extra (M7); every import path
here raises a clear error if missing.
"""

from __future__ import annotations

import numpy as np


def _require_ps():
    try:
        import pterasoftware as ps
    except ImportError as exc:
        raise ImportError(
            "the UVLM backend needs the optional dependency: "
            "pip install pterasoftware  (or pip install fwmav-sim[uvlm])"
        ) from exc
    return ps


def _pair_wings(vehicle) -> list[tuple]:
    """Group per-side wings into (right, left) pairs with identical geometry."""
    pairs = []
    for w in vehicle.wings:
        if w.side != "right":
            continue
        left = next((l for l in vehicle.wings if l.side == "left"
                     and abs(l.span.value - w.span.value) < 1e-12
                     and abs(l.root_chord.value - w.root_chord.value) < 1e-12
                     and abs(l.tip_chord.value - w.tip_chord.value) < 1e-12), None)
        if left is None:
            raise ValueError(
                f"UVLM adapter: wing {w.name!r} has no identical left mirror "
                "(asymmetric pairs unsupported — recorded limitation)")
        pairs.append((w, left))
    return pairs


def _build_airplane(ps, vehicle):
    """Paired symmetric ptera Wings (see module docstring for the mapping)."""
    wings = []
    for w, _left in _pair_wings(vehicle):
        hinge = np.asarray(w.hinge_location.value, dtype=float)
        gamma = w.dihedral_angle.value if w.dihedral_angle is not None else 0.0
        wings.append(ps.Wing(
            wing_cross_sections=[
                ps.WingCrossSection(
                    num_spanwise_panels=vehicle.blade_elements.n_elements,
                    chord=w.root_chord.value,
                    Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing="cosine",
                    airfoil=ps.Airfoil(name="naca0008"),  # thin ~flat plate stand-in
                ),
                ps.WingCrossSection(
                    num_spanwise_panels=None,
                    chord=w.tip_chord.value,
                    Lp_Wcsp_Lpp=(0.0, w.span.value, 0.0),
                    angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                    control_surface_symmetry_type="symmetric",
                    control_surface_hinge_point=0.75,
                    control_surface_deflection=0.0,
                    spanwise_spacing=None,
                    airfoil=ps.Airfoil(name="naca0008"),
                ),
            ],
            name=w.name,
            Ler_Gs_Cgs=tuple(hinge),
            angles_Gs_to_Wn_ixyz=(np.degrees(-gamma), 0.0, 0.0),
            symmetric=True,
            mirror_only=False,
            symmetryNormal_G=(0.0, 1.0, 0.0),
            symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
            num_chordwise_panels=4,
            chordwise_spacing="uniform",
        ))
    return ps.Airplane(wings=wings, name=vehicle.name)


def _wing_movement(ps, ap_wing, kin):
    """One ptera WingMovement from our prescribed kinematics (mapping in the
    module docstring)."""
    f = kin.flap_frequency.value
    delta_s = kin.stroke_phase.value if kin.stroke_phase is not None else 0.0
    psi_amp = (kin.deviation_amplitude.value
               if kin.deviation_amplitude is not None else 0.0)
    delta_psi = (kin.deviation_phase.value
                 if kin.deviation_phase is not None else 0.0)
    period = 1.0 / f
    amps = (np.degrees(kin.flap_amplitude.value),
            np.degrees(kin.pitch_amplitude.value),
            np.degrees(psi_amp))
    return ps.WingMovement(
        base_wing=ap_wing,
        wing_cross_section_movements=[
            ps.WingCrossSectionMovement(
                base_wing_cross_section=wcs,
                ampLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                periodLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
                phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                ampAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                periodAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
                phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
            )
            for wcs in ap_wing.wing_cross_sections
        ],
        ampLer_Gs_Cgs=(0.0, 0.0, 0.0),
        periodLer_Gs_Cgs=(0.0, 0.0, 0.0),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"),
        phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=amps,
        # their validation: zero amplitude requires zero period AND zero phase
        periodAngles_Gs_to_Wn_ixyz=tuple(period if a != 0.0 else 0.0
                                         for a in amps),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=tuple(
            0.0 if a == 0.0 else ph
            for a, ph in zip(amps, (np.degrees(delta_s) + 90.0,
                                    np.degrees(kin.pitch_phase.value + delta_s),
                                    np.degrees(delta_psi + delta_s)),
                             strict=True)),
    )


def run_uvlm_fixed_body(vehicle, kinematics_map: dict, env,
                        n_wingbeats: int, steps_per_wingbeat: int) -> dict:
    """Prescribed-kinematics UVLM solve; returns per-step force history
    (wind axes ~= NED at hover) and the ptera solver for inspection."""
    ps = _require_ps()
    airplane = _build_airplane(ps, vehicle)
    pairs = _pair_wings(vehicle)

    # ptera splits each symmetric Wing into base + reflected; both move with
    # the right wing's kinematics (the mirror handles the geometry)
    wing_movements = []
    for w, _left in pairs:
        kin = kinematics_map[w.name]
        base = next(w2 for w2 in airplane.wings if w2.name == w.name)
        refl = next((w2 for w2 in airplane.wings if w2.name == f"Reflected {w.name}"),
                    None)  # no split when the hinge sits on the symmetry plane
        wing_movements.append(_wing_movement(ps, base, kin))
        if refl is not None:
            wing_movements.append(_wing_movement(ps, refl, kin))
    airplane_movement = ps.AirplaneMovement(
        base_airplane=airplane, wing_movements=wing_movements,
        ampCg_GP1_CgP1=(0.0, 0.0, 0.0), periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1=("sine", "sine", "sine"), phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
    )
    f = next(iter(kinematics_map.values())).flap_frequency.value
    op_movement = ps.OperatingPointMovement(
        base_operating_point=ps.OperatingPoint(
            rho=env.rho, nu=env.nu, vCg__E=max(1e-3, float(np.linalg.norm(env.v_air_W))),
        ))
    f = next(iter(kinematics_map.values())).flap_frequency.value
    all_static = all(
        (kin.flap_amplitude.value == 0.0 and kin.pitch_amplitude.value == 0.0
         and (kin.deviation_amplitude is None or kin.deviation_amplitude.value == 0.0))
        for kin in kinematics_map.values())
    movement = ps.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=op_movement,
        delta_time=1.0 / (f * steps_per_wingbeat),
        # static motion (all amplitudes zero) takes num_steps, not num_cycles
        num_steps=(n_wingbeats * steps_per_wingbeat if all_static else None),
        num_cycles=(None if all_static else n_wingbeats),
    )
    problem = ps.UnsteadyProblem(movement=movement)
    solver = ps.UnsteadyRingVortexLatticeMethodSolver(unsteady_problem=problem)
    solver.run(prescribed_wake=True, show_progress=False)

    sps = solver.unsteady_problem.steady_problems
    forces_W = np.array([sp.airplanes[0].forces_W for sp in sps])
    moments_W = np.array([sp.airplanes[0].moments_W_CgP1 for sp in sps])
    t = np.arange(len(sps)) * movement.delta_time
    return {
        "t": t, "forces_W": forces_W, "moments_W": moments_W,
        "solver": solver, "note": "wind axes; at hover v~0 these are ~NED",
    }
