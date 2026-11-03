# Milestone Report Template

Per project brief §14.10. One report per milestone: `docs/milestone_reports/M{N}.md`.
Fill every section with actual results; delete nothing. "Actual numeric results"
means numbers copied from the test/verification run, not descriptions of intent.

## M{N} — {title}

- **Date:** YYYY-MM-DD
- **Software revision:** `{git rev-parse HEAD}`
- **Environment:** Python {x.y.z}, OS, relevant package versions (`pip freeze` excerpt)

### 1. Implemented capabilities

What works now, mapped to the milestone's task list. Distinguish *tested*
capabilities from *written but untested* code.

### 2. Tests run and actual results

| Check | Command | Result (actual numbers) |
|---|---|---|
| unit tests | `pytest tests/unit -q` | e.g. `14 passed, 0 failed, 21 skipped (stubs)` |
| … | | |

Include convergence tables, tolerance checks and timing where the milestone
requires them. Copy the numbers verbatim from the run.

### 3. Known failures and limitations

Honest list. Include every skipped test whose skip is not a planned stub,
every `PENDING` validation item, and every regime where output is unsupported.

### 4. Deferred items

What was explicitly pushed to later milestones, with the target milestone.

### 5. Reproduction

```bash
# exact commands from a clean checkout
git checkout <hash>
python -m venv .venv && source .venv/Scripts/activate
pip install -e .[dev]
...
```

### 6. Acceptance self-check

The milestone's acceptance criteria, verbatim from the stage file, each marked
PASS / FAIL / PENDING with evidence.
