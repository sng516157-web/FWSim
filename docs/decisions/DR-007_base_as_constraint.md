# DR-007 — Base constraint is experiment data, not code path

**Status: accepted (M6.75, 2026-11-05).**

## Context

Both reference platforms (flappy; He's FlapSim, arXiv:2610.40902) treat the
base constraint as a scenario option on one simulation: flappy flies free by
default; He's runs fixed-base / gimbal / wind / release as switches in a
manifest. FWSim instead had separate experiment *kinds* — `fixed_body`,
`free_body`, `suspended_pendulum` — so "same vehicle and kinematics, but
free instead of clamped" required a new experiment file, and the launch
transient (clamped until T, then released) was inexpressible.

## Decision

The constraint becomes experiment data:

```yaml
base: {mode: fixed | free | pivot, pivot_body_point: [x, z] | null,
       release_time: T | null}
```

- `fixed` — clamped rig (today's `fixed_body`; its fast direct-stepping
  runner is retained as the same physics at ud = 0).
- `free` — today's `free_body` articulated plant.
- `pivot` — today's `suspended_pendulum` constraint model.
- `release_time` — clamp the base DoFs until T, then integrate free
  (launch transient).

Existing kinds become backward-compatible aliases mapping onto (plant,
constraint) pairs; old experiment YAMLs run unchanged. A `base` block may
override the kind's implied constraint — the override is recorded in the
run record's validity section, never silent.

## Consequences

- Every experiment gains the free-body option by editing one block — the
  user's stated requirement.
- Free-body records gain the per-term per-wing force channels the rig
  runner has (the plant already computed them; they are now surfaced), so
  rig-vs-flight comparisons use one channel schema.
- The unified runner layer is the one M7's UVLM adapter plugs into — this
  decision is why M6.75 lands before M7.
