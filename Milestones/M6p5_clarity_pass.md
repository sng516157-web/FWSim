# M6.5 — Clarity Pass (analysis, cleanup, approachability)

> **Stage prompt for Kimi Code.** Read first:
> - `../docs/literature/open_source_sim_survey.md` — survey + code-level notes
> - `../docs/milestone_reports/M6.md`, `../docs/user_guide.md`
> - The two external references: [flappy](https://github.com/purdue-biorobotics/flappy), [He's FlapSim](https://arxiv.org/abs/2610.40902)
>
> **Goal:** close the approachability gap with the two simple external
> projects without surrendering the verification/provenance machinery that
> makes this project trustworthy. Documented plan first (this file), then
> execute in order.

## Design questions every change must answer (user's rubric)

1. **Why should this exist?** (a panel, a module, a file — one sentence)
2. **How does it integrate?** (single data flow: Load → Edit → Run → Record →
   Views; no hidden couplings)
3. **How is it implemented without confusing the user?** (sensible default
   visible first; machinery behind it)

---

## Task 0 — Comparative analysis document (`docs/literature/flappy_flapsim_strengths.md`)

Exactly what the two projects excel at, with evidence, and the concrete
items we copy (each mapped to a landing spot in this repo):

- **flappy excels at:** single validated configuration discipline;
  wing-integrated analytic aero (speed + brevity); actuator/sensor/estimator
  realism in the loop; wing-beat-modulation control channels; learn-by-tiny-
  example packaging.
- **He's FlapSim excels at:** block-composable surrogate with ablation
  methodology; rigorous speed-vs-fidelity quantification (we have never
  benchmarked our own runtime); manifest-driven robot description; honest
  metric definitions.
- **We copy:** examples-first teaching (Task 2); compact authoring (Task 3);
  runtime benchmarking entry (backlog); sensor/estimator layer (M8 backlog);
  C_N/d_cp registry entries (pending citation from the Deng series).

## Task 1 — Repo cleanup

- Delete untracked empty junk: `NVIDIA Corporation/`.
- Move one-off M3 debug probes `derivations/probe_{accel,bias,bias2,bias3}.py`
  → `derivations/probes/` (keep: referenced in spatial.py comments; update
  references). Keep `reduced_longitudinal.py` (live SymPy generator),
  `airpulse_wingmass.py`, `m5_screenshots.py`, `_urdf_demo/`.
- Move `reference_rl_README.md` + `reference_rl_flapping.py` →
  `docs/reference/rl_sandbox/` (not imported anywhere; the foil tests embed
  the closed form). Update the one action-plan reference path if needed.
- Gitignore check: `.pytest_cache/`, `.ruff_cache/` already covered.

## Task 2 — `examples/` (flappy/Ptera-style teaching scripts)

Three short, dependency-light scripts a student can read end-to-end:
1. `examples/fixed_body_hover.py` — load vehicle, run a fixed-body hover,
   print cycle-mean forces (~30 lines).
2. `examples/free_body_hover.py` — free-body run, print the trajectory
   summary (~40 lines).
3. `examples/closed_loop_hover.py` — the M6 control loop end-to-end (~50 lines).
Each runs in CI as a smoke test (they must not drift from the API).

## Task 3 — Compact vehicle authoring (`vehicle/compact.py`)

Problem: provenance-wrapped YAML is unreadable for authoring (20 physical
values → 200 lines). Solution: a compact dialect —
`format: compact-v1` at the top of a vehicle YAML; scalars written as plain
values with an optional `source:` at the block level; `expand_compact(dict)`
stamps full provenance (`estimated` default, overridable) and hands the
standard schema. The full verbose format remains the storage/truth format;
compact is authoring sugar only. Tests: compact ↔ expanded equivalence,
round-trip through schema validation.

## Task 4 — `docs/architecture.md` (one-pager)

The four concepts (vehicle / experiment / record / validity), the module
map (kinematics → aero → dynamics → simulation → analysis → ui), the data
flow (YAML → TimeSeries → record → views), and the design rules (rubric
above) with one example each. ≤ 2 pages.

## Task 5 — GUI basic mode

Default window = ribbon + centre (3D/scope) + run panel. Inspector/editor/
records/validity/console docks start hidden behind an **Advanced ▾** toggle
on the ribbon. Every panel gets a one-line header answering "why this
exists" (rubric item 1). No behavioural change beyond visibility — tested
offscreen (default state = basic, toggle reveals docks).

## Acceptance

- `pytest tests/ -q` green; `ruff` clean.
- `python examples/fixed_body_hover.py` runs and prints sensible numbers.
- A compact vehicle YAML loads and equals its expanded form.
- `git status` clean of junk; docs updated (user guide links architecture.md).
