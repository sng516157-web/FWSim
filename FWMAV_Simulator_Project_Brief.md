# FWMAV Simulation and Control Workbench

## Implementation Brief for Kimi Code

**Status:** Initial project specification  
**Purpose:** Guide implementation of a validated, extensible simulation workbench for flapping-wing micro aerial vehicles (FWMAVs), with particular support for unconventional control mechanisms such as centre-of-gravity (CoG) shifting through an actuated abdomen.

This document is self-contained. It defines the problem, modelling decisions, software scope, implementation order and acceptance criteria. It is a behavioural and scientific contract, not a prescription for an existing code architecture.

---

## 1. Executive Summary

Build a **configurable FWMAV flight-dynamics and control workbench**, not a general MATLAB clone and not another general-purpose robotics simulator.

The core workflow is:

```text
Vehicle definition
→ Wing and mechanism kinematics
→ Aerodynamic and actuator models
→ Coupled flight dynamics
→ Time-resolved simulation
→ Visualization, analysis and experimental comparison
```

The initial technical differentiator is support for **morphing-body and articulated-mass control**, especially a servo-actuated abdomen that changes the instantaneous centre of mass, inertia, force-to-CoM lever arms and internal reaction torques.

The initial scientific promise is deliberately limited:

> Predict and inspect time-resolved FWMAV forces and flight dynamics using explicit reduced-order models, then quantify their agreement with analytical references and experimental data.

Do not describe the software as a universal or high-fidelity FWMAV simulator unless subsequent evidence supports that claim.

---

## 2. Why This Project Should Exist

Existing general simulators can provide rigid-body dynamics, visualization and contact physics. FWMAV researchers still commonly need to assemble their own:

- Wing geometry and flapping kinematics
- Reduced-order unsteady aerodynamic models
- Actuator and mechanism models
- Wingbeat-synchronous data processing
- Flight-dynamics analysis
- Experimental validation workflows

This project should own that missing workflow.

The intended users are:

1. FWMAV researchers comparing vehicle or control designs.
2. Students learning flapping-wing forces and flight dynamics.
3. Engineers developing controllers before hardware testing.
4. Researchers investigating unconventional effectors such as articulated abdomens, moving ballast and morphing bodies.

The project is initially best understood as a **research tool and open technical platform**, not a product with an assumed large commercial market.

---

## 3. Product Principles

### 3.1 Physics before interface

A polished viewer around untrustworthy dynamics is only a demonstration. Implement analytical tests and validation alongside every physical subsystem.

### 3.2 Explicit fidelity levels

Support replaceable reduced-order models. Do not hide model assumptions inside a single “simulate” action.

### 3.3 Time-resolved simulation is authoritative

Resolve forces and motion within each wingbeat. Cycle-averaged forces, trim models and controller-oriented representations should be derived from the time-resolved result, not maintained as unrelated calculations.

### 3.4 Novel control mechanisms are first-class citizens

The vehicle cannot be assumed to be one rigid body with a fixed centre of mass. The architecture must support articulated bodies, time-varying mass properties and actuator reaction forces.

### 3.5 Validation is part of the product

Every result must be traceable to its vehicle definition, model choices, parameter sources, solver settings and software version. Experimental overlays must distinguish calibration from validation data.

### 3.6 Reuse proven infrastructure selectively

Use numerical and visualization libraries where useful. Do not spend the first milestones rebuilding generic plotting, linear algebra, ODE solvers or a general-purpose block-diagram environment.

---

## 4. Initial Scope and Non-Goals

### 4.1 MVP physical scope

The first complete version should support:

- A thorax and two independently configurable wings.
- Rigid wing geometry discretized into spanwise blade elements.
- Prescribed flapping and wing-pitch kinematics.
- Extended quasi-steady blade-element aerodynamics.
- Separately inspectable translational, rotational and added-mass force terms.
- Fixed-body force simulation.
- Six-degree-of-freedom free-body flight simulation.
- An articulated abdomen with pitch and roll joints.
- Time-varying whole-vehicle CoM and inertia.
- Prescribed abdomen motion initially; actuator-driven motion after the multibody plant is tested.
- Still air and uniform wind.
- Instantaneous and wingbeat-averaged outputs.
- Analytical regression tests.
- Experimental time-series import and comparison.
- Time-step and blade-element-count convergence studies.

### 4.2 Explicit MVP non-goals

Do not implement or claim:

- Computational fluid dynamics (CFD).
- Full fluid–structure interaction (FSI).
- Flexible-wing predictive aerodynamics.
- Wake capture or clap-and-fling physics.
- General forward-flight envelope validity.
- A complete motor, gearbox and linkage model in the first vertical slice.
- A MATLAB-language interpreter.
- A general Simulink-compatible block library.
- Photorealistic rendering.
- Automatic sim-to-real controller transfer.
- Experimental validation without actual experimental data.

### 4.3 Deferred capabilities

Potential later additions:

- Prescribed or measured wing deformation.
- Reduced-order structural dynamics.
- Mechanism-generated rather than prescribed wing kinematics.
- Motor and transmission dynamics.
- Periodic trim and Floquet stability analysis.
- Alternative aerodynamic models, including UVLM or lookup-table models.
- Hardware-in-the-loop execution.
- A visual controller/block editor.
- Higher-fidelity external CFD or FSI coupling.

---

## 5. Physical Difficulties and Chosen Solutions

## 5.1 Unsteady aerodynamics

### Problem

A flapping wing experiences rapidly changing translational velocity, pitch rate and acceleration. Relevant mechanisms can include translational circulation and drag, rotational circulation and drag, added-mass forces, leading-edge vortex behaviour, circulatory lag and wake interactions.

A basic blade-element model can match selected average forces while missing instantaneous peaks or phase behaviour.

### Chosen solution

Use a modular aerodynamic interface with an **extended quasi-steady blade-element model** as the default:

\[
\mathbf F_{\mathrm{wing}}
=
\sum_{i=1}^{N}
\left(
\mathbf F_{\mathrm{translation},i}
+\mathbf F_{\mathrm{rotation},i}
+\mathbf F_{\mathrm{added\ mass},i}
\right).
\]

Each blade element must receive:

- Position and orientation.
- Local chord and spanwise area.
- Velocity relative to air.
- Angular velocity and acceleration.
- Local angle of attack.
- Reynolds number and atmospheric properties.

Return force components separately as well as their resultant. Do not present the default model as capturing wake capture, clap-and-fling or arbitrary separated flow.

## 5.2 Geometry, kinematics and frame conventions

### Problem

Small changes in amplitude, pitch timing, stroke-plane angle or pitching-axis location can significantly alter local velocity and angle of attack. Incorrect transforms can produce visually plausible but physically wrong forces.

### Chosen solution

Create an independent kinematics layer with explicit:

- World, body, stroke-plane, wing-root and blade-element frames.
- Units, axis directions, angle signs and rotation order.
- Position, velocity and acceleration calculations.
- Prescribed flapping, pitching and optional deviation kinematics.
- Left/right symmetry handling.
- Visualization of frames, local velocity, angle of attack and force directions.

All aerodynamic models must consume the same kinematic-state representation.

## 5.3 Wing flexibility and FSI

### Problem

Wing bending and twisting alter local angle of attack and aerodynamic loading. Full FSI is too expensive and complex for the MVP.

### Chosen solution

Define a structural-fidelity ladder:

1. Rigid wing.
2. Imported or prescribed deformation.
3. Reduced-order flexible wing.
4. External high-fidelity FSI coupling.

Implement only Level 1 initially. The result must clearly state that wing deformation is excluded.

## 5.4 Wing–body and abdomen–body coupling

### Problem

Body motion changes wing velocity relative to air. Wing and abdomen motion change mass distribution and can induce reaction forces and torques. A single rigid-body model with fixed mass properties cannot represent these effects.

### Chosen solution

Use generalized multibody dynamics:

\[
M(q)\ddot q + C(q,\dot q)\dot q + G(q)
=
\tau_{\mathrm{actuator}}
+
J_a(q)^T\mathbf F_{\mathrm{aero}}.
\]

Support distinct operating modes:

- Fixed-body force analysis.
- Prescribed-body-motion analysis.
- Free-body flight.
- Suspended or otherwise constrained test configurations.

Do not treat these as identical plants.

The relative airflow at each element must include body motion:

\[
\mathbf v_{\mathrm{relative}}
=
\mathbf v_{\mathrm{wing\ element}}
-
\mathbf v_{\mathrm{air}}.
\]

## 5.5 Multiple time scales

### Problem

The solver must resolve events within each wingbeat while also simulating manoeuvres over many cycles. Cycle averaging can remove useful transient and oscillatory dynamics.

### Chosen solution

- Resolve each wingbeat using configurable substeps and solver tolerances.
- Store instantaneous and wingbeat-averaged quantities.
- Derive averaged models from the same time-resolved run.
- Do not assume periodic steady state; detect or test for it.
- Defer periodic trim and Floquet analysis until the base dynamics are tested.

## 5.6 Actuation and power

### Problem

Prescribed motion assumes an actuator can perfectly execute the requested trajectory. Real systems have torque, speed, friction, power and motion-profile limits.

### Chosen solution

Support two explicit modes:

1. Prescribed kinematics for aerodynamic analysis.
2. Actuator-driven dynamics for control and feasibility analysis.

The actuator interface should eventually expose torque, speed, saturation, reaction torque and estimated power. Do not silently use ideal prescribed motion when the user selects actuator-driven mode.

## 5.7 Numerical reliability

### Problem

Rapid reversals, low inertia and optional structural dynamics make results sensitive to integration settings.

### Chosen solution

- Specify resolution as steps per wingbeat as well as timestep in seconds.
- Use established numerical integrators with configurable tolerances.
- Represent attitude with quaternions or rotation matrices.
- Detect non-finite states, excessive motion and solver failure.
- Save solver settings and random seeds with each experiment.
- Require timestep and blade-element convergence studies.
- Never label a run converged merely because it reached its end time.

## 5.8 Parameter uncertainty and identification

### Problem

Aerodynamic coefficients, inertias and mechanism properties may be uncertain. Fitting many parameters to one dataset can produce non-predictive agreement.

### Chosen solution

For every parameter, record where applicable:

- Value and unit.
- Source: measured, literature-derived, estimated or fitted.
- Uncertainty or valid range.
- Calibration dataset.
- Applicable flight regime.

Support sensitivity analysis and selected-parameter fitting. Keep calibration and validation datasets separate.

## 5.9 Experimental comparison

### Problem

Load-cell data can contain inertial forces and structural vibration. Free-flight reconstruction is sensitive to tracking and filtering. Simulation and experiment must represent equivalent measured quantities.

### Chosen solution

Provide a native comparison workflow:

1. Import experimental force, motion or actuator data.
2. Synchronize wingbeat phase.
3. Apply documented filtering consistently where appropriate.
4. Compare instantaneous values, cycle averages, spectra and trajectories.
5. Display error metrics and residual time series.
6. Record uncertainty when available.
7. Separate calibration and validation runs.

---

## 6. CoG-Shift and Actuated-Abdomen Control

## 6.1 Why this requires a multibody plant

CoG-shift control is not merely another aerodynamic coefficient. An actuated abdomen can produce control through:

1. Changing the CoM relative to aerodynamic and thrust application points.
2. Changing the whole-vehicle inertia tensor.
3. Applying internal reaction torques.
4. Producing inertial forces during rapid acceleration.
5. Generating direct aerodynamic forces if the abdomen acts as a rudder.

The vehicle model must therefore include an upper body, articulated abdomen, component masses and joints. Abdomen pitch and roll must have their own configuration, velocity and acceleration states.

## 6.2 Instantaneous mass properties

Compute the whole-vehicle centre of mass from the current configuration:

\[
\mathbf r_{\mathrm{CoM}}(q)
=
\frac{\sum_i m_i\mathbf r_i(q)}
{\sum_i m_i}.
\]

Compute the inertia about that point:

\[
I_{\mathrm{CoM}}(q)
=
\sum_i
\left[
R_i I_i R_i^T
+
m_i
\left(
\|\mathbf d_i\|^2\mathbf I
-\mathbf d_i\mathbf d_i^T
\right)
\right],
\qquad
\mathbf d_i=\mathbf r_i-\mathbf r_{\mathrm{CoM}}.
\]

These values must derive from actual body geometry and configuration, not a visualization-only CoM marker.

## 6.3 Correct control-moment interpretation

Gravity has no resultant moment about the instantaneous centre of mass. In untethered flight, moving the CoG does not by itself create a “gravity torque about the CoM.”

The relevant moment is:

\[
\boldsymbol{\tau}_{\mathrm{control}}
=
(\mathbf r_T-\mathbf r_{\mathrm{CoM}})
\times \mathbf F_T
+
\sum_j
(\mathbf r_{a,j}-\mathbf r_{\mathrm{CoM}})
\times \mathbf F_{a,j}
+
\sum_j \mathbf M_{a,j}
+
\boldsymbol{\tau}_{\mathrm{inertial/reaction}}.
\]

The simulator must be able to show the lever arms and force vectors underlying this equation.

A suspended-string experiment has a physical pivot and pendulum dynamics. It is not equivalent to free flight and must be represented by a separate constraint model.

## 6.4 Abdomen actuator model

Implement in stages:

1. Prescribed abdomen angle.
2. Motion-profile-limited abdomen angle.
3. Torque-driven joint with damping and limits.
4. Servo dynamics with reaction torque and power estimates.

Record commanded and actual position, velocity, torque and saturation.

Slow CoG shifts and rapid abdomen motions must remain distinguishable: the former primarily changes force geometry, while the latter can also exploit dynamic coupling.

## 6.5 Fidelity levels for abdomen-control studies

### Level 1 — Reduced analytical model

- Longitudinal motion only.
- Thorax and abdomen as concentrated masses.
- Small perturbations near hover.
- Cycle-averaged wing forces.
- Linearized state-space representation.

Use for analytical regression tests and early PID/LQR experiments.

### Level 2 — Nonlinear articulated model

- Six-DoF thorax.
- Abdomen pitch and roll.
- Nonlinear multibody dynamics.
- Time-resolved wing forces.
- Time-varying CoM and inertia.
- Joint and actuator limits.

This is the minimum intended model for free-flight CoG-shift manoeuvre studies.

### Level 3 — Abdomen aerodynamics and oscillatory control

- Abdomen lift and drag.
- Separate abdomen centre of pressure.
- Wingbeat-synchronized abdomen inputs.
- Optional vibrational control experiments.
- Experimentally identified corrections.

## 6.6 Mechanism-isolation experiments

Provide switches or model variants for:

| Case | Purpose |
|---|---|
| CoM motion disabled | Baseline response |
| Geometry-only CoG shift | Quasi-static lever-arm contribution |
| Full inertial coupling | Abdomen acceleration contribution |
| Abdomen aerodynamics only | Aerodynamic-rudder contribution |
| All enabled effects | Combined prediction |

These comparisons are more informative than fitting one combined response curve.

---

## 7. General Control-Effector Architecture

Do not hard-code abdomen control as the only special case. Use the common pipeline:

```text
Control command
→ Actuator dynamics
→ Mechanism configuration
→ Aerodynamic forces, joint torques and mass-property changes
→ Vehicle dynamics
```

The architecture should eventually accommodate:

- Wing-amplitude modulation.
- Wing-frequency modulation.
- Stroke-plane modulation.
- Asymmetric wing pitching.
- Abdomen CoG shifting.
- Moving internal ballast.
- Articulated aerodynamic tails.
- Conventional control surfaces.

Enable comparisons using common metrics:

- Recovery angle and settling time.
- Tracking error.
- Control effort and actuator saturation.
- Mechanical/electrical energy where modelled.
- Wingbeat-level attitude disturbance.
- Sensitivity to mass and aerodynamic uncertainty.

A general block-diagram editor is optional and deferred. Programmatic controller composition is sufficient for the MVP.

---

## 8. Software Architecture Contract

The exact internal architecture may be chosen during implementation, but preserve these modelling boundaries:

```text
VehicleDefinition
├── Geometry and mass properties
├── Bodies, joints and coordinate frames
├── Wing kinematics
├── Structural model
├── Actuator models
└── Aerodynamic-model selection

Simulation
├── Environment
├── Test constraints
├── Coupled dynamics
├── Numerical solver
└── Controller

ExperimentRecord
├── Vehicle and model configuration
├── Parameter provenance
├── Solver configuration
├── Time-resolved outputs
├── Derived wingbeat metrics
└── Validation and software metadata
```

Required conceptual interfaces:

- `WingKinematics`
- `AerodynamicModel`
- `StructuralModel`
- `ActuatorModel`
- `ControlEffector`
- `FlightDynamics`
- `EnvironmentModel`
- `TestConstraint`
- `Controller`
- `ExperimentalDataset`

These are responsibilities, not mandatory class names.

### 8.1 Recommended implementation strategy

Start with a small, testable scientific core:

- Python for orchestration and initial modelling.
- NumPy/SciPy for arrays, integration and analysis.
- A clear separation between equations of motion, force models and visualization.
- A lightweight 2D/3D viewer only after basic numerical tests pass.
- A general multibody library may be evaluated for the articulated model, but do not build a general physics engine from scratch.

Candidate references or dependencies to evaluate—not automatic commitments:

- **Drake:** articulated multibody dynamics, systems and control.
- **Pinocchio:** rigid-body algorithms, analytical derivatives and model import.
- **bdsim:** Python block-diagram execution if controller composition later needs it.
- **MeshCat or Viser:** browser-based 3D visualization.
- **OpenModelica/URDFModelica:** reference for acausal multibody modelling and URDF conversion.

Choose the backend through a short technical spike. Do not integrate several dynamics engines simultaneously.

### 8.2 Data and reproducibility

Use human-readable, versioned project definitions where practical. A saved experiment must include enough information to reproduce:

- Vehicle geometry and inertial parameters.
- Model and submodule selections.
- Initial state and environment.
- Controller and actuator commands.
- Solver configuration.
- Parameter provenance.
- Output schema and software revision.

Plots shown in the UI must derive from actual simulation output. Never hard-code demonstration curves as physical results.

---

## 9. User Workflow

The first end-to-end workflow should be:

1. Select a reference vehicle.
2. Inspect geometry, coordinate frames and mass distribution.
3. Configure wing kinematics and aerodynamic model.
4. Run a fixed-body wingbeat simulation.
5. Inspect instantaneous and averaged forces.
6. Enable the articulated abdomen.
7. Run a suspended or constrained abdomen-actuation test.
8. Compare against analytical or experimental data.
9. Progress to free-body simulation.
10. Save and reload the complete experiment.
11. Repeat headlessly and obtain identical outputs within solver tolerance.

The first UI should prioritize:

- Vehicle/tree inspector.
- Parameter editor with units and valid ranges.
- 3D vehicle and force visualization.
- CoM, aerodynamic centre and lever-arm overlays.
- Time-series plots.
- Experiment configuration and run status.
- Assumption/validity panel.
- Experimental-data comparison.

Do not begin with a dashboard full of disconnected controls.

---

## 10. Milestones and Acceptance Criteria

## Milestone 0 — Modelling specification and test plan

Deliver:

- Coordinate-frame and sign-convention document.
- State definitions and equations for the reduced analytical model.
- Parameter schema with units.
- List of model assumptions.
- Regression-test plan.

Acceptance:

- A second developer can determine every frame, state and force direction without reading implementation code.
- No ambiguous use of pitch, flap angle, thrust direction or CoM reference point.

## Milestone 1 — Fixed-body aerodynamic vertical slice

Deliver:

- One rigid wing with prescribed kinematics.
- Blade-element discretization.
- Initial quasi-steady force components.
- Instantaneous and averaged force output.
- Basic visualization and plots.

Acceptance:

- Zero motion produces the expected zero/no-flapping baseline.
- Mirrored configurations produce the expected mirrored forces.
- Reversing a defined input changes the appropriate quantity according to the documented convention.
- Increasing blade count and temporal resolution produces a reported convergence trend.
- Every force component can be inspected independently.

## Milestone 2 — Two-wing vehicle and analytical abdomen model

Deliver:

- Two-wing configuration.
- Longitudinal thorax–abdomen model.
- Instantaneous CoM and inertia calculations.
- Passive and actuated pendulum cases.
- Comparison against the reduced analytical reference.

Acceptance:

- CoM and inertia agree with independently calculated simple configurations.
- Gravity moment about the instantaneous CoM is zero in the appropriate free-body calculation.
- Suspended constraints are explicitly represented.
- The reduced model reproduces analytical limiting cases within documented numerical tolerance.
- No claim of free-flight validation is made from pendulum agreement.

## Milestone 3 — Nonlinear articulated free-body simulation

Deliver:

- Six-DoF body motion.
- Abdomen pitch and roll configuration.
- Coupled aerodynamic and multibody dynamics.
- Prescribed and motion-limited abdomen inputs.
- Mechanism-isolation runs.

Acceptance:

- Fixed-body and constrained modes remain available as separate test configurations.
- The simulator exposes force-to-CoM lever arms and internal joint torques.
- Results are repeatable after save/reload and headless execution.
- Timestep refinement does not materially change the reported metrics within the declared tolerance.
- Solver failures stop the run visibly rather than producing plausible-looking partial output.

## Milestone 4 — Validation and experiment comparison

Deliver:

- Experimental CSV or equivalent time-series importer.
- Time/phase alignment.
- Filtering metadata.
- Error metrics and residual plots.
- Calibration/validation dataset labels.

Acceptance:

- Imported and simulated quantities have matching units and definitions.
- The report distinguishes aerodynamic, inertial and measured resultant forces where relevant.
- No validation badge or claim appears without a real dataset and defined comparison metric.
- If reference data are unavailable, the corresponding validation task remains explicitly pending.

## Milestone 5 — Integrated workbench

Deliver:

- Vehicle and experiment editor.
- 3D and time-series visualization.
- CoM/aerodynamic-centre overlays.
- Assumption and model-validity display.
- Reproducible experiment save/load.
- Headless execution of the same project.

Acceptance:

- A new user can execute the complete reference workflow without editing source code.
- UI and headless runs produce equivalent numerical outputs.
- Every demo plot is generated from the current model, not stored presentation data.

---

## 11. Mandatory Test Categories

Implement tests incrementally rather than postponing them until the UI exists.

### 11.1 Unit tests

- Rotation and frame transforms.
- Kinematic derivatives.
- Blade-element velocity and angle calculations.
- Individual aerodynamic terms.
- CoM and parallel-axis inertia calculations.
- Joint constraints and actuator limits.
- Wingbeat averaging.

### 11.2 Analytical and limiting-case tests

- Stationary wing.
- Uniform translation.
- Symmetric left/right flapping.
- Zero-abdomen-motion baseline.
- Concentrated-mass pendulum configurations.
- Small-angle linearized model.
- Free-body gravity moment about instantaneous CoM.

### 11.3 Numerical tests

- Timestep refinement.
- Blade-element refinement.
- Solver tolerance sensitivity.
- Quaternion/attitude normalization.
- Save/reload reproducibility.
- Headless/UI equivalence.

### 11.4 Scientific validation

- Published analytical or numerical comparison where licensing and data availability permit.
- Suspended-abdomen experiment comparison.
- Separate free-flight validation when data become available.
- Mechanism-ablation comparisons.

Record failed comparisons honestly. Do not tune a model on validation data and continue to label that dataset as validation.

---

## 12. Reference Vehicle: Bio-Inspired Abdomen FWMAV

Use the uploaded paper as the initial domain reference:

**Jiaxun Leng, Yuk Wa Steven Ng, and Shun Tang. “Novel Flapping Wing Micro Aerial Vehicle Design Featuring a Bio-inspired Abdomen Mechanism.” IMAV 2024.**

Relevant concepts from the paper:

- X-wing upper body and actuated abdomen.
- Abdomen pitch and roll control.
- Electronics concentrated in the abdomen.
- Two-concentrated-mass longitudinal analytical model.
- Euler–Lagrange derivation and small-angle linearization.
- Suspended-string transient experiment.
- High-speed-camera pitch tracking.
- Webots simulation using experimentally informed unsteady lift.
- Explicit limitation that suspended experiments do not capture the full free-flight envelope.

Implementation requirements:

- Reproduce the reduced analytical case before extending it.
- Treat suspension, aerodynamic-force scaling and other experimental assumptions explicitly.
- Do not infer three-dimensional free-flight validity from the reported suspended-test agreement.
- Request or locate original parameter values and time-series data before attempting numerical reproduction; if unavailable, implement the test structure and mark data-dependent validation as pending.

---

## 13. Priorities

### P0 — Required for a credible first release

- Documented frames and conventions.
- Fixed-body rigid-wing force simulation.
- Time-resolved quasi-steady aerodynamics.
- Analytical thorax–abdomen model.
- Correct instantaneous CoM and inertia.
- Nonlinear articulated dynamics.
- Constrained versus free-body test modes.
- Reproducible experiment records.
- Numerical regression and convergence tests.
- Honest assumption and validation reporting.

### P1 — Adds major research value

- Experimental-data import and residual analysis.
- Actuator dynamics and reaction torque.
- Mechanism-isolation experiments.
- Controller interface and strategy comparison.
- Sensitivity analysis.
- Integrated visualization and parameter editor.
- Periodic trim tools.

### P2 — Expand only after validation and use justify them

- Flexible wings and FSI coupling.
- Higher-fidelity aerodynamic models.
- Full drivetrain and power modelling.
- Hardware-in-the-loop execution.
- General visual block editor.
- Broad vehicle-template library.
- CFD integration.

---

## 14. Execution Contract for Kimi Code

When implementing this project:

1. Start with Milestone 0 and the smallest numerical vertical slice; do not begin by generating a large UI.
2. Preserve the distinction between physical models, numerical solvers, experiment definitions and visualization.
3. Reuse established numerical/multibody libraries where they reduce risk, but evaluate them before committing.
4. Write tests alongside each physical subsystem.
5. Never fabricate parameters, reference results, experimental data or validation claims.
6. Clearly label placeholders, unvalidated models and unsupported operating regimes.
7. Do not silently reduce scope. Report missing capabilities and their effect on results.
8. Do not claim convergence, validation or reproducibility without running the corresponding checks.
9. Prefer a small number of complete, tested milestones over many partially connected features.
10. At the end of each milestone, report:
   - Implemented capabilities.
   - Tests run and actual results.
   - Known failures and limitations.
   - Deferred items.
   - Exact commands needed to reproduce the result.
11. If a library or backend blocks progress, document the attempted evidence and propose a focused alternative rather than hiding the problem.
12. Do not introduce CFD, FSI, a general scripting language or a general block editor before the P0 dynamics and validation path works.

---

## 15. First Implementation Task

The first coding task should be intentionally narrow:

> Create the project foundation, document coordinate conventions, implement a configurable two-body longitudinal thorax–abdomen model, calculate instantaneous CoM and inertia, and verify passive and small-angle behaviour against independently derived analytical references. Expose a minimal programmatic run interface and numerical plots. Do not add a general UI or free-flight aerodynamics in this task.

This gives the project a trustworthy dynamics kernel and an immediate connection to the novel CoG-shift control problem. The fixed-body aerodynamic vertical slice can then be developed in parallel or immediately afterwards without confusing pendulum validation with aerodynamic validation.

---

## 16. Technical References and Existing Projects

These are starting references, not endorsements of every modelling assumption:

- Drake: https://github.com/RobotLocomotion/drake
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- bdsim: https://github.com/petercorke/bdsim
- URDFModelica: https://github.com/DLR-RM/urdfmodelica
- OpenModelica: https://openmodelica.org/
- MeshCat: https://github.com/meshcat-dev/meshcat
- Viser: https://github.com/nerfstudio-project/viser

Research directions to consult when implementing or extending models:

- Quasi-steady flapping-wing aerodynamics and experimental validation.
- Added-mass, rotational-circulation and unsteady-lag models.
- FWMAV free-flight system identification.
- Wing flexibility and reduced-order fluid–structure modelling.
- Periodic flight dynamics and wingbeat-averaged control.
- Abdominal control in insects and articulated-body aerial vehicles.

Cite the exact equations and coefficient sources used in implementation. Do not implement aerodynamic equations from memory without recording their source and assumptions.
