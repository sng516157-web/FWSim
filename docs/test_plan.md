# Test Plan — Brief §11 Mapping

**Status: frozen in M0; stubs created for every row.** Every bullet of brief
§11.1–11.4 maps to a concrete file and test function. Functions marked
*(stub)* exist as skipped placeholders; `pytest --collect-only` lists them
all (M0 acceptance). A stub is replaced by a real test in the milestone shown
in the last column — skipping it silently is not allowed; the skip reason
names the milestone.

## §11.1 Unit tests

| Brief §11.1 bullet | Test function | File | Milestone |
|---|---|---|---|
| Rotation and frame transforms | `test_quat_matrix_roundtrip` | `tests/unit/test_transforms.py` | M1 |
| | `test_rotation_composition_associative` | `tests/unit/test_transforms.py` | M1 |
| | `test_rotation_derivative_finite_difference` | `tests/unit/test_transforms.py` | M1 |
| Kinematic derivatives | `test_prescribed_kinematics_matches_finite_difference` | `tests/unit/test_kinematics.py` | M1 |
| | `test_reversed_phi_amp_documented_sign_change` | `tests/unit/test_kinematics.py` | M1 |
| Blade-element velocity and angle calculations | `test_element_velocity_hand_computed` | `tests/unit/test_blade_element.py` | M1 |
| | `test_element_aoa_hand_computed` | `tests/unit/test_blade_element.py` | M1 |
| Individual aerodynamic terms | `test_translational_term_at_0_45_90deg` | `tests/unit/test_aero_terms.py` | M1 |
| | `test_rotational_term_hand_computed` | `tests/unit/test_aero_terms.py` | M1 |
| | `test_added_mass_term_hand_computed` | `tests/unit/test_aero_terms.py` | M1 |
| CoM and parallel-axis inertia calculations | `test_com_hand_computed` | `tests/unit/test_mass_properties.py` | M2 |
| | `test_inertia_parallel_axis_hand_computed` | `tests/unit/test_mass_properties.py` | M2 |
| Joint constraints and actuator limits | `test_joint_limit_clamping` | `tests/unit/test_mass_properties.py` | M2 |
| | `test_profile_rate_limit` | `tests/unit/test_actuation.py` | M3 |
| | `test_torque_saturation_recorded` | `tests/unit/test_actuation.py` | M3 |
| Wingbeat averaging | `test_wingbeat_average_constant_signal` | `tests/unit/test_averaging.py` | M1 |
| | `test_phase_segmentation` | `tests/unit/test_averaging.py` | M1 |
| *(M0) parameter schema* | `test_schema.py` (real, 12 tests) | `tests/unit/test_schema.py` | M0 ✔ |
| *(M0) frame registry* | `test_frames_registry.py` (real, 4 tests) | `tests/unit/test_frames_registry.py` | M0 ✔ |

## §11.2 Analytical and limiting-case tests

| Brief §11.2 bullet | Test function | File | Milestone |
|---|---|---|---|
| Stationary wing | `test_stationary_wing_zero_force` | `tests/analytical/test_fixed_body.py` | M1 |
| Uniform translation | `test_uniform_translation_steady_drag_sign` | `tests/analytical/test_fixed_body.py` | M1 |
| Symmetric left/right flapping | `test_mirrored_wings_mirrored_forces` | `tests/analytical/test_symmetry.py` | M1 |
| | `test_symmetric_flapping_zero_roll_yaw_moment` | `tests/analytical/test_symmetry.py` | M2 |
| Zero-abdomen-motion baseline | `test_zero_abdomen_motion_baseline` | `tests/analytical/test_reduced_model.py` | M2 |
| | `test_reduced_model_matches_sympy_reference` | `tests/analytical/test_reduced_model.py` | M2 |
| Concentrated-mass pendulum configurations | `test_passive_pendulum_frequency` | `tests/analytical/test_pendulum.py` | M2 |
| | `test_actuated_pendulum_response` | `tests/analytical/test_pendulum.py` | M2 |
| Small-angle linearized model | `test_linearized_vs_numerical_jacobian` | `tests/analytical/test_reduced_model.py` | M2 |
| Free-body gravity moment about instantaneous CoM | `test_gravity_moment_about_com_zero` | `tests/analytical/test_mass_properties.py` | M2 |

## §11.3 Numerical tests

| Brief §11.3 bullet | Test function | File | Milestone |
|---|---|---|---|
| Timestep refinement | `test_timestep_refinement` (real) | `tests/numerical/test_convergence.py` | M1 ✔ |
| Blade-element refinement | `test_blade_element_refinement` (real) | `tests/numerical/test_convergence.py` | M1 ✔ |
| *(M2)* pendulum energy drift | `test_frictionless_pendulum_energy_drift` (real) | `tests/numerical/test_pendulum_energy.py` | M2 ✔ |
| *(M2)* actuated-case timestep refinement | `test_actuated_pendulum_timestep_refinement` (real) | `tests/numerical/test_pendulum_energy.py` | M2 ✔ |
| *(M2 §2.4a)* RL-sandbox foil | `test_foil_rhs_equivalence`, `test_foil_hover_equilibrium`, `test_foil_pitch_response` (real) | `tests/analytical/test_foil_rl_sandbox.py` | M2 ✔ |
| Solver tolerance sensitivity | `test_solver_tolerance_sensitivity` | `tests/numerical/test_convergence.py` | M3 |
| Quaternion/attitude normalization | `test_quaternion_norm_post_step` | `tests/numerical/test_attitude.py` | M3 |
| Save/reload reproducibility | `test_save_reload_rerun_reproduces` | `tests/numerical/test_reproducibility.py` | M3 |
| Repeat-run determinism (M1 stage §1.4) | `test_repeat_run_determinism` (real) | `tests/numerical/test_reproducibility.py` | M1 ✔ |
| Headless/UI equivalence | `test_headless_ui_equivalence` | `tests/numerical/test_ui_equivalence.py` | M5 |
| *(M3) solver-failure behaviour* | `test_nan_forces_fail_visibly` | `tests/numerical/test_failure.py` | M3 |

## §11.4 Scientific validation

| Brief §11.4 bullet | Test function | File | Milestone |
|---|---|---|---|
| Published analytical/numerical comparison | `test_leng2024_suspended_comparison` (skip: awaiting original data, brief §12) | `tests/validation/test_leng2024.py` | M4 (PENDING data) |
| Suspended-abdomen experiment comparison | `test_suspended_abdomen_comparison` (same dataset dependency) | `tests/validation/test_leng2024.py` | M4 (PENDING data) |
| Free-flight validation when data become available | `test_free_flight_comparison` (skip: no data) | `tests/validation/test_free_flight.py` | M4+ (PENDING data) |
| Mechanism-ablation comparisons | `test_mechanism_isolation_cases` | `tests/validation/test_mechanism_isolation.py` | M3 |
| *(pipeline proof without validation claim)* | `test_synthetic_dataset_roundtrip` | `tests/validation/test_synthetic_pipeline.py` | M4 |

## Rules

1. A skipped stub must name its milestone in `reason=`; a real failure is
   never converted to a skip to make a milestone pass.
2. Data-dependent validation stays `pytest.mark.skip(reason="awaiting
   original data")` until the Leng et al. (IMAV 2024) parameter table and
   time series arrive (action plan §6, open item 1).
3. Failed validation comparisons are recorded honestly in the milestone
   report (brief §11.4); a dataset used for fitting is thereafter labelled
   calibration, never validation.
