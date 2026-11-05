# What flappy and He's FlapSim Excel At — and What We Copy (M6.5 Task 0)

Companion to `open_source_sim_survey.md`. This document is specific: each
strength is stated with evidence (file/line or paper section), why it works,
and the concrete item copied into FWSim (or deferred, with where it lands).

## flappy ([purdue-biorobotics/flappy](https://github.com/purdue-biorobotics/flappy), ICRA 2019)

### S1 — Single validated configuration discipline
The repo ships exactly one vehicle (their hummingbird) with rig-validated
aero coefficients and one working hover stack. Nothing speculative is
presented as usable. *Why it works:* a newcomer runs `test.py --model_type=PID`
and sees a working hover in minutes; there is no way to accidentally use
placeholder physics without noticing.
*Copy:* our `data_status` + provenance already encode this intent; we add
**examples/ that use only literature-sourced configurations** (M6.5 Task 2)
and keep estimated-placeholder vehicles visibly labelled in the GUI.

### S2 — Wing-integrated analytic aerodynamics (`envs/Wing.cpp`, ~250 lines)
One normal force per wing from analytic spanwise polynomial integrals of the
quasi-steady pressure, with planform moments (r00…r33) and their own
rig-fitted `C_N(α)` and `d_cp(α)` curves. *Why it works:* fully analytic
(closed-form integrals instead of per-element loops), cheap enough to embed
in RL training loops, and the single effective-AoA approximation is validated
on their rig.
*Copy (deferred):* not a replacement for our per-element model (finer where
AoA varies along the span). Their `C_N`/`d_cp` fits are **registry candidates
pending citation** (Deng-lab series; we need the exact paper before
inclusion — added to M7 backlog). The quadratic pitch-rate damping term
cross-checks our FD-derived `dMy/dωy` (M6).

### S3 — Actuator, sensor and estimator realism in the loop
`Actuator.cpp` (voltage-driven motor dynamics — kinematics *emerge*), plus
`sensor.py`/`sensor_fusion.py` (IMU model + complementary-filter estimator).
Controllers see estimated, not true, state. *Why it works:* controllers that
survive this loop survive hardware; ours currently get perfect state.
*Copy (deferred):* **sensor/estimator simulation layer — M8 theme.**

### S4 — Wing-beat-modulation control channels
Four channels (thrust / differential / mean / split-cycle) decouple the
controller from the mechanism. *Copy:* already in M7 Task 4.

### S5 — Learn-by-tiny-example packaging
`test.py`, `test_simple.py`, `train.py` — each < 100 lines, each doing one
visible thing end-to-end. *Copy:* **M6.5 Task 2 verbatim.**

## He's FlapSim ([arXiv:2610.40902](https://arxiv.org/abs/2610.40902), 2026)

### S6 — Block-composable surrogate with ablation methodology
The surrogate is mean + optional blocks (clock sync, inflow, geometry,
harmonics), and the paper reports an *ablation per block* (Table 3/Fig. 4),
so every block's contribution is measured, not asserted. *Why it works:*
complexity is admitted only with evidence.
*Copy:* our unsteady terms (M6 T7) and future aero blocks should be
reported ablation-style: a per-term on/off comparison table in the run's
report. **M7 backlog: ablation report generator.**

### S7 — Speed-vs-fidelity quantification discipline
Every model reports fidelity AND CPU speedup (67×, 68×...) — the trade is
the headline metric. We have never benchmarked our own runtime.
*Copy:* **M7 backlog: runtime benchmark** (per experiment kind, per fidelity
tier, printed in the M7 report; `time.perf_counter` around the dispatch).

### S8 — Manifest-driven, composable robot description
Robot manifests compose geometry + aero plugin + pipeline. Similar intent
to our vehicle/experiment split; theirs is flatter (one file).
*Copy:* our **compact authoring** (M6.5 Task 3) closes most of the gap;
the two-file split stays (provenance needs it).

### S9 — Honest metric definitions
"Fidelity" is defined precisely (filtered simulator-vs-simulator match, not
physical validation) and its limits stated in the abstract's shadow.
*Copy:* keep our validity layer; cite their bounded-claims style in reports
(exactly what our M4 honest-failure note already does).

### S10 — Their documented timing traps
The 21-substep clock-slip analysis (float-time drift) is a rare honest
report of an integration-timing defect. *Copy:* our stepped integrator
already uses integer sample grids; note in the integrator docs why
(records their trap as rationale).

## Copy list (consolidated, with landing spot)

| # | Item | Lands |
|---|---|---|
| 1 | examples-first teaching (S5) | M6.5 Task 2 |
| 2 | compact vehicle authoring (S8) | M6.5 Task 3 |
| 3 | GUI basic mode (S1 spirit) | M6.5 Task 5 |
| 4 | C_N/d_cp registry entries (S2) | M7 backlog (needs citation) |
| 5 | ablation report generator (S6) | M7 backlog |
| 6 | runtime benchmark (S7) | M7 backlog |
| 7 | sensor/estimator layer (S3) | M8 theme |
| 8 | actuator dynamics (S3) | M8 theme (A-004 relaxation) |
