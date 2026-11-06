# Milestones — Stage Prompts for Kimi Code

One self-contained prompt per project stage. Hand Kimi Code the file for the current stage only; each file points to the contract documents it must read first.

| File | Stage | Gate to start | Est. effort |
|---|---|---|---|
| [M0_modelling_specification.md](M0_modelling_specification.md) | Modelling specification and test plan | — | 3–5 days |
| [M1_fixed_body_aero.md](M1_fixed_body_aero.md) | Fixed-body aerodynamic vertical slice | M0 accepted | 1.5–2.5 weeks |
| [M2_two_wing_analytical_abdomen.md](M2_two_wing_analytical_abdomen.md) | Two-wing vehicle + analytical abdomen model | M1 accepted | 1.5–2 weeks |
| [M3_articulated_free_body.md](M3_articulated_free_body.md) | Nonlinear articulated free-body simulation | M2 accepted | 3–4 weeks |
| [M4_validation_comparison.md](M4_validation_comparison.md) | Validation and experiment comparison | M3 accepted | 1.5–2 weeks |
| [M5_integrated_workbench.md](M5_integrated_workbench.md) | Integrated workbench (UI) | M4 accepted | 3–4 weeks |
| [M6_usability_and_physics_fidelity.md](M6_usability_and_physics_fidelity.md) | Usability + physics fidelity (unsteady terms, control loop, sysid, CI) | M5 accepted | — |
| [M6p5_clarity_pass.md](M6p5_clarity_pass.md) | Clarity pass (examples, compact authoring, GUI basic mode) | M6 accepted | — |
| [M6p75_unified_base_constraint.md](M6p75_unified_base_constraint.md) | Unified base constraint (fixed/free/release as data) | M6.5 accepted | — |
| [M7_fidelity_ladder_and_interfaces.md](M7_fidelity_ladder_and_interfaces.md) | Fidelity ladder (UVLM tier, passive pitch) + learning interfaces | M6.75 accepted | — |
| [M8_node_workbench.md](M8_node_workbench.md) | Node workbench (pipeline executor, block library, node editor, platform builder) | M7 accepted | — |
| M9 — workbench redesign | Live-first workbench: mechanical tree + control DAG, planform editor, Qt-native GL, live tuning — see `../docs/design/M9_workbench_redesign.md` | M8 accepted | — |

## Rules of engagement (all stages)

1. **Milestone gating:** do not start stage N+1 until stage N's acceptance commands pass and its milestone report (`docs/milestone_reports/M{N}.md`) is written (brief §14.9, §14.10).
2. **Contract hierarchy:** `../FWMAV_Simulator_Project_Brief.md` > `../FWMAV_Milestone_Action_Plan.md` > the stage file. Conflicts: stop and flag, never choose silently.
3. **Route A (confirmed 2026-11-03):** pure-Python portable core; in-house spatial-vector multibody dynamics (DR-001); PySide6 desktop UI (DR-002). No Drake, no Pinocchio, no conda in the runtime path.
4. **Honesty requirements:** no fabricated parameters, reference results, experimental data, or validation claims; placeholders and pending items explicitly labelled (brief §14.5–14.8).

## Open items that affect execution (action plan §6)

- ~~Leng, Ng & Tang (IMAV 2024) parameter table and suspended-test time series~~ — **closed 2026-11-04: co-author confirms no such dataset exists.** The tests stay skipped with that reason; AirPulse (arXiv:2602.06811) is the literature-grade reference vehicle instead.
- Conda availability for the optional dev-only Pinocchio cross-check in M3.0 (WSL2 is an acceptable substitute).
- Repository hosting and license.
