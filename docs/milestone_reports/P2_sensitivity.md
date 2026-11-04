# P2 — Parameter sensitivity analysis (post-M5 queued item 2)

- **Date:** 2026-11-04
- **Software revision:** `612da8a` + this change
- **Environment:** Python 3.14.7, Windows (Git Bash)

Question answered: which physical parameters actually move the simulation
outputs — i.e. what is worth measuring carefully on a real vehicle.

## 1. Implemented capabilities

| Capability | Where | Tested |
|---|---|---|
| One-at-a-time central-difference sensitivity with one-sided fallback at domain boundaries (schema-invalid perturbations) | `analysis/sensitivity.py` | yes |
| Dot-path parameter addressing into vehicle/kinematics/environment YAML trees | `analysis/sensitivity.py` | yes |
| Kind-aware metrics (fixed body: cycle-mean forces/power per wing + total; dynamic kinds: final position norm, peak excursion; FAILED runs recorded as NaN) | `analysis/sensitivity.py` | yes |
| `fwmav-sim sensitivity` CLI; ranked Markdown + full JSON record | `cli.py` | yes |
| In-process reruns without saving records | `simulation/run.py` `dispatch_experiment` (refactor; `run_experiment` behaviour unchanged) | yes (reproducibility suite) |

## 2. Tests run and actual results

| Check | Command | Result |
|---|---|---|
| analytic exponents | `pytest tests/numerical/test_sensitivity.py -q` | **3 passed**: force ∝ rho (S = 1.0 ± 1e-8), force ∝ f² (S = 2.0 ± 1e-8), power ∝ f³ (S = 3 + δ² central-difference bias, exact), body mass absent from fixed-body forces (S = 0 ± 1e-10) |
| auto parameter set | (in suite) | covers bodies/wings/environment/kinematics; massless carriers excluded |
| demo: m1_fixed_body_hover, 18 parameters auto | `fwmav-sim sensitivity experiments/m1_fixed_body_hover.yaml` | record `outputs/20261104_211114_sensitivity_m1_fixed_body_hover` |

Demo result (metric: total cycle-mean vertical force; |S| ranked):

| rank | parameter | S |
|---|---|---|
| 1 | flap_frequency | 2.00 |
| 2 | flap_amplitude | 1.91 |
| 3 | wing span (both) | 1.44 |
| 4 | air density | 1.00 |
| 5 | pitch_phase | 0.48 |
| 6 | pitch_amplitude | 0.36 |
| 7 | tip/root chord | 0.31 / 0.25 |
| 8 | pitch_axis_fraction | d/dp only (zero base) |
| — | body masses, viscosity | 0 |

## 3. Known limitations

- OAT central differences: no interaction terms (not a Sobol/Morris analysis);
  5% default step, configurable via `--rel`.
- Normalized S is undefined at zero base value (e.g. `pitch_offset = 0`); the
  report shows the unnormalized derivative there.
- Free-text `uncertainty_or_range` provenance strings are not machine-readable;
  the analysis uses uniform relative steps, not the documented ranges.
- Dynamic-kind metrics (position norms) are crude; a trim/Floquet sensitivity
  would be the follow-up for control design.

## 4. Reproduction

```bash
pytest tests/numerical/test_sensitivity.py -q
fwmav-sim sensitivity experiments/m1_fixed_body_hover.yaml
```
