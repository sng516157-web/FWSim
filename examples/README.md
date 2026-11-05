# Examples

Small end-to-end scripts — the fastest way to learn what FWSim does
(flappy/Ptera-style teaching: each file is one complete, readable story).

| Script | Shows | Runtime |
|---|---|---|
| `fixed_body_hover.py` | the rig: clamped body, prescribed wingbeats, cycle-mean forces | ~10 s |
| `free_body_hover.py` | the flight: 6-DoF body released under flapping wings | ~1 min |
| `closed_loop_hover.py` | the design loop: trim → linearize → LQR → regulate | ~1 min |

Run any of them from the repo root after `pip install -e .[dev]`:

```bash
python examples/fixed_body_hover.py
```

Each script prints what to look at and which validity warnings apply.
For the concepts (vehicle / experiment / record / validity) see
`docs/architecture.md`; for the full panel tour see `docs/user_guide.md`.
