# DR-004 — Vehicle geometry and kinematics derived from CAD (long-term direction)

**Status:** design direction, recorded 2026-11-04 (Steven). **Milestone of effect:** M3+.

## Decision / direction

The reference vehicle's geometry and kinematic layout should ultimately be
**derived from the user's CAD model** (mass properties, hinge locations,
wing-pair topology, stroke-plane orientation), not hand-maintained as
placeholder YAML. The X-wing configuration (two wing pairs per side,
anti-phase fore/aft kinematics) is the driving case.

## Implications for the current code path

- The schema already supports multiple wings per side (`wings` list); per-pair
  kinematic phase offsets need a small extension (kinematics are currently
  global per experiment).
- Placeholder `estimated` values in `reference_vehicles/bio_abdomen_fwmav.yaml`
  remain until the CAD-derived (or paper-supplied) parameters land;
  `data_status` stays `pending_original_parameters`.
- A CAD import path (e.g., URDF/mesh + mass properties from CAD export) is a
  later work item; no dependency is added before that spike (action plan §5).

## Immediate work item (with M3)

X-wing reference-vehicle support: four wings (fore/aft pairs), per-pair
phase offsets, X-wing stroke-plane treatment. Clap-and-fling aerodynamics
remain excluded (A-002) even with the X-wing geometry in place.
