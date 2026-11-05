# M6.75 — Unified Base Constraint & Platform Hardening

> **Stage prompt for Kimi Code.** Read first:
> - `../docs/literature/flappy_flapsim_strengths.md` — the copy list (S1–S10)
> - `../docs/architecture.md`, `../docs/assumptions.md`
> - `../Milestones/M7_fidelity_ladder_and_interfaces.md` (sequencing: this stage lands FIRST — the UVLM adapter plugs into the unified runner)

**Goal:** (a) the base constraint (fixed / free / pivot / release) becomes an
**option on every experiment**, not a separate experiment kind — the flappy /
He's-FlapSim pattern of one simulation with scenario switches; (b) clear the
concrete backlog items from the strengths doc that don't need M7's UVLM.

**User's framing (verbatim):** "all simulation should have the option for
free body, rather than simply locked in place."

## Task 0 — DR-007: base as constraint (decision record)

One simulation pipeline; the constraint is data, not code path:
- `base: fixed` — clamped (today's fixed_body rig; keeps its fast
  direct-stepping runner as the same physics at ud = 0)
- `base: free` — today's free_body articulated plant
- `base: pivot` — today's suspended_pendulum constraint model
- `release_time: T` — clamp until T, then free (launch transient; new)
Existing kinds (`fixed_body`, `free_body`, `suspended_pendulum`) become
backward-compatible aliases mapping onto (plant, constraint) pairs — old
YAMLs keep working unchanged.

## Task 1 — Schema: `base` block

`ExperimentDefinition.base: BaseConstraintSpec | None` with
`{mode: fixed|free|pivot, pivot_body_point?, release_time?}`. Default None =
legacy kind behaviour. Old kinds map: fixed_body → (fixed), free_body →
(free), suspended_pendulum → (pivot). New kind not required; `base` may
override the kind's implied constraint (documented precedence, warning in
the record when it overrides).

## Task 2 — Unified force recording

free_body records gain the per-term per-wing channels the fixed_body runner
has (`F_trans`/`F_rot`/`F_am`/`F_unsteady` sums per wing — the plant already
computes them; surface them). Base-fixed runs through the plant emit the
same channel schema as the fast fixed_body runner (drift test: forces agree
to solver tolerance for an identical case).

## Task 3 — `release_time`: clamp → release

Clamp base DoFs (prescribed zero) until t = release_time, then integrate
free. The classic FWMAV launch transient, impossible to express today.
Tests: forces before release match the fixed-body runner bit-near-exactly;
linear/angular momentum are continuous across the release instant; a
divergence-free run on the bio vehicle. Experiment
`experiments/m6p75_launch_transient.yaml` + `examples/launch_transient.py`.

## Task 4 — Registry: flappy C_N / d_cp entries (citation-gated)

Add `purdue_hummingbird_fit` to `aero/registry.py`
(`C_N = 1.8 sin2α cosα + 1.95 sinα − 1.5 cos2α sinα`) — **blocked until the
exact Deng-lab citation is supplied** (the fit appears in flappy's
`Wing.cpp`; the paper source must be confirmed, user to provide or approve
the citation we find). Optional `d_cp(α)` aero-moment block: pitching moment
per element from the local centre of pressure — new physics for us (the M4
moment channels currently carry no CP model), gated behind its own flag.

## Task 5 — Runtime benchmark (`fwmav-sim benchmark`)

He's-FlapSim-style speed quantification: wall time per experiment kind +
per fidelity tier (quasi-steady vs +unsteady terms vs +wing inertia),
printed table + JSON to outputs/. Reported in the milestone report (first
runtime numbers for the project).

## Task 6 — Ablation report generator

He's-style ablation table: for a given experiment, run it with each optional
physics block toggled (unsteady terms, wing inertia, aero damping) and
tabulate the change in cycle-mean forces — `analysis/ablation.py` +
`fwmav-sim ablate <experiment.yaml>` → markdown + JSON. Teaches users what
each block contributes (S6).

## Acceptance

- `pytest tests/ -q` green, ruff clean; every old experiment YAML runs
  unchanged (reproducibility tests).
- `fwmav-sim run experiments/m6p75_launch_transient.yaml` produces a record
  with continuous momentum across release.
- `fwmav-sim benchmark` prints the runtime table.
- `fwmav-sim ablate experiments/m6_unsteady_airpulse.yaml` prints the
  ablation table.
