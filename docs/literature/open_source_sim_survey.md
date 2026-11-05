# Open-Source Flapping-Wing Simulators — Survey & Method Distillation (2026-11-05)

> **Name note (2026-11-05):** this project was renamed **FWSim** (from
> "FlapSim") after the independent publication of *FlapSim: A Composable
> MuJoCo Simulation Platform and Reduced-Order Aerodynamic Model Hierarchy
> for Flapping-Wing Robotics* (Mingshan He, [arXiv:2610.40902](https://arxiv.org/abs/2610.40902),
> [github.com/MingshanHe/FlapSim](https://github.com/MingshanHe/FlapSim)).
> The package/CLI remain `fwmav_sim` / `fwmav-sim`. He's FlapSim is also
> highly relevant PRIOR ART: its plain cycle-averaged surrogate scores only
> 0.66 mean wrench fidelity vs its own blade-element reference (0.24 in
> wind, 0.49 during frequency steps) — independent confirmation of our M6
> finding that mean forces alone are insufficient. Its remedy (inflow/gust
> coupling blocks, phase-locked 1f–6f harmonic reconstruction, ridge-fitted
> surrogate at 68x speedup) generalizes the same idea as our
> `dynamics/aero_damping.py`; their wind/gust derivative blocks are the
> blueprint for extending our damping to wind axes. Also noted: they
> excluded rotational-lift and added-mass plugins for producing unphysical
> mean loads, while our quasi-steady core includes both — a cross-check
> candidate (M7 backlog).

Compiled for M7 planning. Verdict up front: the user's thesis holds — generic
robotics sims (Gazebo/MuJoCo) ship no unsteady flapping aero; every serious
effort builds a custom aero core. The field's methods cluster into four
fidelity tiers, and FWSim already sits in tiers 1–2 with a distinctive
honesty/provenance layer nobody else has.

## The landscape

### 1. Flappy Hummingbird — Purdue Biorobotics (ICRA 2019)
[purdue-biorobotics/flappy](https://github.com/purdue-biorobotics/flappy) · [paper](https://arxiv.org/abs/1902.09628)
- **Stack:** DART multibody engine + custom quasi-steady blade-element aero
  (validated against measured rig forces) + TensorFlow-era RL.
- **Methods worth taking:**
  - *Control-channel abstraction:* the plant is driven through helicopter-style
    wing-beat modulation channels — thrust (voltage amplitude), roll
    (differential), pitch (mean), yaw (split-cycle) — not raw kinematic angles.
    Controllers see a clean 4-channel interface regardless of mechanism.
  - *Gym environment* with baseline cascading-PID and adaptive robust
    controllers — the community-standard way to publish a control testbed.
- **Weaknesses:** heavy native dependency (DART), stale (TF1.x), no provenance
  discipline, geometry hard-coded to their robot.

### 2. Ptera Software — C. Urban et al. (validated 2021, actively maintained)
[camUrban/PteraSoftware](https://github.com/camUrban/PteraSoftware) · [validation thesis](https://repository.library.washu.edu/mems500/149/)
- **Stack:** pure Python (Numba JIT, parallel), pip-installable, PyVista viz.
- **Aero:** ring **UVLM** — steady horseshoe/ring VLM and unsteady ring UVLM
  with **fixed or free wake**, **vortex aging** against numerical instability,
  loads via Kutta-Joukowski or Katz & Plotkin pressure integration, ground
  effect via method of images, airfoil camber-line geometry from the UIUC
  database, cosine panel spacing.
- **Methods worth taking:**
  - UVLM is the validated, cheap next fidelity step above quasi-steady blade
    element: wake structure and induced effects emerge instead of being
    bolted on (our wake-capture/clap-and-fling reduced terms are crude
    proxies for what a free wake gives naturally).
  - **Trim search tools** (`analyze_steady_trim`/`analyze_unsteady_trim`).
  - **Free-flight coupling**: their UVLM drives a 6-DoF body (they use
    MuJoCo; we have our own verified dynamics — same pattern).
  - **Aeroelastic hook**: torsional-spring compliance per wing cross-section.
  - **Validated against experimental flapping-wing data** with the case
    published — the community-standard credibility move.
  - JSON save/reload of solved cases.

### 3. FLOWUnsteady — BYU FLOW Lab
[byuflowlab/FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady)
- **Stack:** Julia; reformulated vortex particle method (rVPM) high end.
- **Method worth taking:** the **explicit variable-fidelity ladder** — VLM,
  strip theory, blade elements, 3D panel, rVPM in one suite with the
  fidelity/cost trade exposed per run. Mirrors our Level 1/2/3 structure;
  M7 should formalize the ladder the same way (and rVPM stays out of scope).

### 4. flybody — DeepMind + Janelia (Nature 2025)
[TuragaLab/flybody](https://github.com/TuragaLab/flybody) · [paper](https://www.nature.com/articles/s41586-025-09029-4)
- **Stack:** MuJoCo + custom low-Re flapping-wing aerodynamic force model
  (quasi-steady blade-element style with added-mass terms — architecturally
  the same choice as FWSim) + deep RL locomotion tasks.
- **Method worth taking:** even DeepMind, with MuJoCo in-house, models fly
  aerodynamics as quasi-steady blade element — external validation of our
  core aero architecture. Their RL task packaging (Gymnasium-style) is the
  current API standard (supersedes flappy's old Gym).

### 5. Gazebo/ROS flapping sims (various, e.g. DpooL2000)
- Custom plugins that "fake the physics" (their words): generic engines have
  no unsteady aero, so forces are injected ad hoc. No methods worth taking;
  included as evidence for the project's premise.

### 6. FlapKine (2024)
- Kinematics/aero visualization tool; our GUI supersedes it.

## Distilled methods → what M7 should adopt

| Method | Source | Why it matters for FWSim |
|---|---|---|
| Free-wake UVLM as the next aero tier | Ptera | Captures wake structure, induced velocity, clap effects from first principles instead of reduced proxies; validated public cases exist to test against |
| Wrap, don't rewrite | Ptera (pip) | PteraSoftware is pip-installable pure Python: an adapter backend inherits their validation and maintenance, at ~10% the cost of a home UVLM |
| Torsional-spring passive wing pitch | Ptera aeroelastic + AirPulse reality | AirPulse pitch is PASSIVE; our prescribed-sinusoid placeholder is the largest known aero-input error (M4) |
| Wing-beat-modulation control channels | flappy | Thrust/roll/pitch/yaw as control inputs decouple controller design from mechanism detail |
| Gymnasium RL wrapper | flybody, flappy | The community interface for the control work the platform invites |
| Trim search as a tool | Ptera | Our trim is reduced-model only; unsteady trim search is the practical need |
| Fidelity ladder made explicit | FLOWUnsteady | One suite, user picks the tier per run, validity panel reports the tier |

## What FWSim already has that none of them do

Provenance on every parameter; the assumption register + per-run validity
section; bit-identical reproducibility; sensitivity analysis; sysid with
calibration/validation label discipline; URDF import; a verified
(non-MuJoCo) articulated dynamics core. M7 should preserve these as the
differentiators while adopting the methods above.

## Code-level notes: flappy's Wing.cpp (added 2026-11-05, second look)

Read of `flappy/envs/Wing.cpp` (their entire aero core, ~250 lines):

- **Wing-integrated (not blade-element) quasi-steady:** one normal force per
  wing via analytic spanwise polynomial integrals — the element velocity is
  linear in span (`U_i1·s + U_i0`), so `F_N = ½ρ c C_N(α_cp)(a_U2 R³r22 +
  a_U1 R²r11 + a_U0 R r00)` with planform moments r00/r11/r22/r33, AoA
  evaluated at a characteristic station `r_cp = R·r33/r22`. Much cheaper and
  fully analytic; cruder (single effective AoA per wing). Ours is
  per-element — finer where AoA varies along the span; theirs is validated
  on their rig. A wing-integrated variant is a legitimate fast force path
  if runtime ever matters.
- **Their own rig-fitted curves** (candidates for our coefficient registry,
  M6 T6): `C_N(α) = 1.8 sin2α cosα + 1.95 sinα − 1.5 cos2α sinα` and a
  center-of-pressure fit `d_cp(α) = 0.46 − 0.332 cosα − 0.037 cos3α −
  0.013 cos5α` (Deng-lab lineage; citation needed from their paper series
  before registry inclusion).
- **Quadratic rotational (pitch-rate) damping moment** `−0.125 ρ |θ̇|θ̇ ·
  5 R c⁴ z_rd` — hardcoded factor 5.0; same physics as our FD-derived
  `dMy/dωy` (M6 aero_damping.py) but as a closed-form term. Cross-check
  candidate: our FD derivative should bracket this.
- **Actuator dynamics** (`Actuator.cpp`): voltage-driven motor model — wing
  kinematics EMERGE from the mechanism rather than being prescribed (an
  A-004 relaxation path).
- **Sensor + estimator simulation** (`sensor.py`, `sensor_fusion.py`): IMU
  model and a complementary-filter estimator inside the control loop —
  controllers see estimated, not true, state. A whole layer FWSim lacks;
  candidate M8 material for realistic control evaluation.
- Robot geometry is URDF + STL meshes (we import URDF mass properties;
  meshes are a viz-only concern).
