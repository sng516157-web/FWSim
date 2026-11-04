# P3 — CAD/URDF import spike (post-M5 queued item 3; DR-004)

- **Date:** 2026-11-04
- **Software revision:** `612da8a` + this change
- **Environment:** Python 3.14.7, Windows (Git Bash), stdlib `xml.etree` only
  (no new dependency, per action plan §5 / DR-004)

## 1. Implemented capabilities

| Capability | Where | Tested |
|---|---|---|
| URDF `<inertial>` → BodySpec (mass, CoM offset, inertia rotated from the inertial frame into the link frame via origin rpy) | `vehicle/urdf_import.py` | yes |
| Revolute/continuous joints about coordinate axes → JointSpec (origin → location, limits) | `vehicle/urdf_import.py` | yes |
| Fixed joints merged into the parent body (parallel-axis folding; nested fixed chains fold child-first; descendant joint origins re-expressed by translation) | `vehicle/urdf_import.py` | yes |
| Clear rejections: prismatic/planar/floating joints, non-coordinate axes, rotated joint origins (rpy), branching link trees | `vehicle/urdf_import.py` | yes |
| `fwmav-sim import-urdf robot.urdf -o vehicle.yaml` CLI (validated YAML skeleton with header comment) | `cli.py` | yes (round-trip reload) |

## 2. Tests run and actual results

| Check | Command | Result |
|---|---|---|
| import unit tests | `pytest tests/unit/test_urdf_import.py -q` | **6 passed** (two-body exactness, fixed-joint merge to 1e-9, inertial rpy rotation, three rejection paths) |
| end-to-end smoke | `fwmav-sim import-urdf derivations/_urdf_demo/demo.urdf -o ... && load_vehicle(...)` | reload validates (`demo_fwmav`, thorax + abdomen + abdomen_pitch) |

## 3. Known limitations (spike scope, recorded in the module docstring)

- Wings do NOT import: URDF has no planform concept — wings are authored
  manually after import. The generated YAML carries `data_status:
  cad_import_spike ...; wings authored manually`.
- Serial chains only (the ArticulatedModel MVP topology); branches rejected.
- Frame convention: URDF frames must already match this project (+x forward,
  +z down) — pre-rotate at CAD export; documented in the generated header.
- Provenance is labelled `estimated` (CAD solid-model mass properties are
  computed, not measured) with the URDF file as reference.

## 4. What a CAD workflow looks like now

1. Export URDF from SolidWorks/Fusion (mass properties from assigned materials).
2. `fwmav-sim import-urdf model.urdf -o reference_vehicles/my_vehicle.yaml`
3. Author `wings:` + `aero:` blocks (planform, hinge, coefficients) manually.
4. Point an experiment YAML at the vehicle and run.

## 5. Reproduction

```bash
pytest tests/unit/test_urdf_import.py -q
fwmav-sim import-urdf derivations/_urdf_demo/demo.urdf -o /tmp/v.yaml
```
