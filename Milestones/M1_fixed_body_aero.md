# Milestone 1 — Fixed-Body Aerodynamic Vertical Slice

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — the contract (especially §3, §5.1, §5.2, §5.7, §10 M1, §11)
> - `../FWMAV_Milestone_Action_Plan.md` — locked stack §1, conventions §3, repo layout §2
> - `docs/conventions/frames.md` and `docs/conventions/notation.md` — **frozen output of M0; do not redefine**
>
> Conflicts: the brief wins; stop and flag. Route A: pure-Python portable core, no Drake/Pinocchio/conda runtime deps.

**Goal:** one rigid wing, prescribed kinematics, blade-element discretization, separable quasi-steady force terms, instantaneous + averaged outputs, convergence evidence (brief §10 M1).
**Estimated effort:** 1.5–2.5 weeks.
**Prerequisite:** M0 accepted (conventions frozen, schema in place).

---

## 1.1 Ground rules for this stage

- Physics before interface (brief §3.1): every subsection below ships with its tests in the same commit.
- **No equations from memory** (brief §16): every aerodynamic formula's docstring cites its source and the assumption IDs (A-…) it relies on.
- Return force components **separately** (translational / rotational / added mass), never only a resultant (brief §5.1).
- This stage is **fixed-body only**: the thorax does not move; there is no abdomen, no free flight, no ODE solve of vehicle dynamics. Prescribed kinematics ⇒ direct time-stepping.
- Every plot derives from actual run output stored in the run record (brief §8.2). No hard-coded demo curves.

## 1.2 Locked stack for this stage

NumPy ≥ 1.26 · SciPy ≥ 1.11 · Pydantic ≥ 2.6 + PyYAML (configs) · Matplotlib ≥ 3.8 (plots) · `meshcat` pip package (3D spike) · pytest. Add nothing else without updating action plan §1.1.

---

## 1.3 Tasks (in order)

### Task 1 — `core/transforms.py`

Rotation matrices from ZYX Euler, quaternion ops (multiply, conjugate, to/from matrix, normalize), angular-velocity ↔ quaternion-rate, frame-chain composition per `frames.md` (`W ← B ← SP ← WR ← BE`).

Tests (`tests/unit/test_transforms.py`): matrix↔quaternion round trip to machine precision; composition `R(a)R(b) == R(ab)`; analytical `Ṙ = R·[ω]×` vs central finite difference `< 1e-6`; the left-wing reflection operator produces mirrored positions/velocities.

### Task 2 — `kinematics/prescribed.py`

Prescribed sinusoidal kinematics with configurable amplitude, frequency, phase, offset:

```python
class PrescribedWingKinematics(BaseModel):
    flap_amplitude: Parameter      # φ_amp [rad]
    flap_frequency: Parameter      # f [Hz]
    pitch_amplitude: Parameter     # θ_w amplitude [rad]
    pitch_phase: Parameter         # lag relative to stroke [rad]
    pitch_offset: Parameter
    deviation_amplitude: Parameter | None
    stroke_plane_angle: Parameter  # β [rad]
```

Support non-sinusoidal pitch profiles via a small Fourier-term list. Provide `state(t) -> (phi, theta_w, psi_w, and time derivatives)` computed **analytically** (no finite differences inside kinematics).

Test: analytical derivatives vs central finite difference `< 1e-6` at 100 random times; reversal of `flap_amplitude` sign changes exactly the quantities documented in `frames.md` (brief §10 M1 acceptance).

### Task 3 — `kinematics/blade_element.py`

Discretize the span into N elements (equal-area option required); per element compute in the BE frame: position, velocity, acceleration, local chord, area, angle of attack, Reynolds number. Everything derived from the frozen transform chain; all aero models consume this single kinematic-state representation (brief §5.2):

```python
@dataclass
class ElementState:
    r_W: np.ndarray        # (3,) element position, world frame
    v_W: np.ndarray        # (3,) element velocity, world frame
    a_W: np.ndarray        # (3,) element acceleration, world frame
    R_W_BE: np.ndarray     # (3,3) element orientation
    omega_W: np.ndarray    # (3,) wing angular velocity
    chord: float
    area: float
    alpha: float           # local angle of attack [rad]
    reynolds: float
```

Tests: one hand-computed element at mid-stroke (position/velocity/AoA to `< 1e-12`); element areas sum to wing planform area; mirrored wings give mirrored states.

### Task 4 — `aero/interface.py`

```python
class AerodynamicModel(Protocol):
    def element_forces(self, state: ElementState, env: Environment) -> ElementForces: ...

@dataclass
class ElementForces:
    F_translational: np.ndarray  # (3,)
    F_rotational: np.ndarray
    F_added_mass: np.ndarray
    @property
    def resultant(self) -> np.ndarray: ...
```

### Task 5 — `aero/quasi_steady.py` (default model)

Extended quasi-steady blade-element model (brief §5.1), all coefficients in YAML with `provenance: literature` and citations:

- **Translational:** Dickinson et al. 1999 Robofly fits — `C_L = 0.225 + 1.58·sin(2.13α − 7.20°)`, `C_D = 1.92 − 1.55·cos(2.04α − 9.82°)`, forces `∝ ½ρv²c·dr`.
- **Rotational circulation:** Sane & Dickinson 2002 form, `∝ ρ·ω̇_pitch·c²·dr` with the published coefficient.
- **Added mass:** flat-plate per-unit-span coefficient `ρ·π·c²/4` acting on normal acceleration (Sedov / Ellington lineage).
- Document what is **not** captured: wake capture, clap-and-fling, arbitrary separated flow (brief §5.1). Wire assumption IDs A-002/A-003 into docstrings and the run record's validity section.

Tests (`tests/unit/test_aero_terms.py`): each term vs hand-computed values at α ∈ {0°, 45°, 90°}; zero relative velocity ⇒ zero force; sign of drag opposes relative motion.

### Task 6 — `environment/`

`still_air.py` and `uniform_wind.py` behind one `EnvironmentModel` protocol. Relative airflow at each element: `v_rel = v_element − v_air` (brief §5.4).

### Task 7 — `simulation/fixed_body.py`

Time-stepped (not ODE-solved) fixed-body runner: configurable **steps per wingbeat** as well as Δt in seconds (brief §5.7); emits a `TimeSeries` containing per-term forces, resultant, per-element forces, and (where modelled) instantaneous power. Wingbeat-cycle segmentation by phase.

### Task 8 — `analysis/averaging.py`

Cycle-averaged forces/moments derived **from the same time-resolved arrays** (brief §3.3); phase-aligned averaging across cycles with per-cycle spread reported.

### Task 9 — `viz/force_plots.py`

Matplotlib: Fx/Fz vs phase with per-term overlay; cycle-averaged bar chart; per-element spanwise force distribution at selected phases. All figures regenerated from the stored run record by `fwmav-sim report <run_dir>`.

### Task 10 — MeshCat spike + minimal scene (`viz/meshcat_scene.py`)

Timeboxed: 1 day. Wing as discretized surface, BE frame triads, per-element velocity and force arrows, CoM marker. **Verify install and rendering on Windows and record it.** If MeshCat fails: write `docs/decisions/DR-00x-meshcat.md` with evidence and fall back to Matplotlib 3D — do not silently drop 3D (brief §14.11).

### Task 11 — Reference vehicle draft

`reference_vehicles/bio_abdomen_fwmav.yaml` (Leng, Ng & Tang IMAV 2024, brief §12). Every unavailable value: `source: estimated`, plus a top-level `data_status: pending_original_parameters` flag. Never invent plausible-looking numbers without the `estimated` label (brief §14.5).

### Task 12 — Convergence study

`experiments/m1_convergence.yaml` driving `python -m fwmav_sim.analysis.convergence`: blade count N ∈ {4, 8, 16, 32, 64} × steps/wingbeat ∈ {50, 100, 200, 400, 800, 1600}; report L2 difference of cycle-averaged forces vs the finest case; store the table in the run record and milestone report. A run is never "converged" because it reached its end time (brief §5.7).

---

## 1.4 Test inventory

**Unit (§11.1):** transforms (Task 1); kinematic derivatives (Task 2); blade-element state (Task 3); each aero term (Task 5); wingbeat averaging on synthetic signals with known means.

**Analytical (§11.2):** stationary wing + still air ⇒ all forces zero (`test_stationary_wing_zero_force`); uniform translation ⇒ steady drag, correct sign and magnitude from C_D(α) alone; mirrored left/right wings ⇒ mirrored forces (`test_mirror_symmetry`); reversed input convention test.

**Numerical (§11.3):** convergence sweep above; repeat-run determinism on fixed inputs.

## 1.5 Acceptance commands

```bash
pytest tests/unit tests/analytical tests/numerical -q
fwmav-sim run experiments/m1_fixed_body_hover.yaml
python -m fwmav_sim.analysis.convergence experiments/m1_convergence.yaml
fwmav-sim report outputs/<run_dir>
```

## 1.6 Milestone report

`docs/milestone_reports/M1.md`: per-term force plots, the convergence table verbatim, mirror/zero-motion test results, coefficient citations, and the explicit statement that wake capture, clap-and-fling and wing deformation are excluded and in which regime the model is claimed to operate.

## 1.7 Explicit non-goals for this stage

Body motion of any kind; abdomen; actuator dynamics; free flight; two-wing interference; UI beyond plots and the MeshCat check; experimental-data comparison.
