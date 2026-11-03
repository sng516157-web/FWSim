# Notation — Symbol-to-Code-Name Binding

**Status: FROZEN (Milestone 0).** Rule (action plan §2): every physical symbol
in code matches this table. Code or a docstring that introduces a physical
symbol not listed here is a **review blocker**. Additions require editing this
table in the same commit.

Conventions (units SI, angles rad, frames per `frames.md`):

- Vectors are column vectors; a left superscript in prose denotes the frame
  of expression, e.g. `ᵂv` = v expressed in W. In code the frame travels in
  the variable name suffix, e.g. `v_W`.
- `R_A_B`: direction-cosine matrix mapping B-coords → A-coords (`frames.md`).
- Overdot = time derivative.

## Wing kinematics

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| φ | `phi` | rad | wing stroke (flapping) angle, + = tip forward, zero lateral (`frames.md` §3) |
| θ_w | `theta_w` | rad | wing pitch (rotation) angle, + = leading edge up, zero = chord in stroke plane |
| ψ_w | `psi_w` | rad | wing deviation angle, + = tip below stroke plane |
| β | `beta` | rad | stroke-plane angle, + = anterior end up (Ry about +y_B) |
| f | `f_flap` | Hz | flapping frequency |
| T_wb = 1/f | `T_wingbeat` | s | wingbeat period |
| φ_amp, θ_amp, ψ_amp | `phi_amp`, `theta_amp`, `psi_amp` | rad | stroke / pitch / deviation amplitudes |
| θ_off | `theta_off` | rad | wing-pitch offset (mean rotation) |
| δ_θ, δ_ψ | `delta_theta`, `delta_psi` | rad | pitch / deviation phase shifts |
| s | `s_span` | m | spanwise station from the wing root, along +y_WR |
| c(s) | `chord` | m | local chord at span station s |
| N | `n_elements` | — | blade-element count per wing |

## Vehicle attitude, frames, transforms

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| ψ, θ, φ_r | `psi`, `theta`, `phi_r` | rad | yaw, pitch (+ nose up), roll (+ = right wing down); ZYX intrinsic |
| q = (w,x,y,z) | `quat` | — | attitude quaternion, scalar-first, Hamilton, normalized |
| ω_B = [p,q,r] | `omega_B` | rad/s | body angular velocity in B (roll, pitch, yaw rates) |
| R_W_B etc. | `R_W_B` | — | DCM mapping B-coords → W-coords; same pattern for all pairs |
| S | `S_mirror` | — | reflection diag(1,−1,1) for left-wing mirroring |
| g | `g` (`G = 9.81`) | m/s² | gravity magnitude; vector `g_vec_W = [0, 0, +9.81]` |

## Mass properties and multibody dynamics

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| q | `q` | varies | generalized coordinate vector |
| v (q̇-space) | `v` | varies | generalized velocity vector |
| τ | `tau` | N·m or N | generalized force vector |
| M(q) | `M_mass` | kg, kg·m² | generalized mass (inertia) matrix |
| C(q,v)v | `C_bias` | N, N·m | Coriolis/centrifugal bias term |
| G(q) | `G_grav` | N, N·m | gravity generalized force |
| J_a(q) | `J_aero` | varies | Jacobian mapping generalized velocity to aero-element motion |
| m_t, m_a | `m_thorax`, `m_abdomen` | kg | thorax / abdomen mass (reduced model: point masses) |
| I_t | `I_thorax` | kg·m² | thorax pitch inertia about its CoM (reduced model) |
| I_CoM(q) | `I_com` | kg·m² | whole-vehicle inertia about the instantaneous CoM |
| r_CoM(q) | `r_com` | m | instantaneous whole-vehicle CoM position |
| r_J | `r_joint` | m | abdomen joint location in B |
| l_a | `l_abdomen` | m | joint-to-abdomen-CoM distance (reduced model) |

## Reduced longitudinal model (see `reduced_model_spec.md`)

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| x, z | `x`, `z` | m | thorax reference point position in W (z positive down) |
| θ | `theta` | rad | thorax pitch, + nose up |
| α_abd | `alpha_abd` | rad | abdomen pitch relative to thorax, + = tip down, zero = aligned |
| x (state) | `state` | — | `x = [x, z, θ, α_abd, ẋ, ż, θ̇, α̇_abd]` |
| T | `thrust` | N | cycle-averaged thrust, along +x_B |
| F_w | `F_wing_avg` | N | cycle-averaged wing-force resultant, applied at wing-root midpoint |
| A, B | `A_lin`, `B_lin` | — | linearized state-space matrices |

## Aerodynamics

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| ρ | `rho` | kg/m³ | air density |
| v_rel | `v_rel` | m/s | element velocity relative to air (`v_element − v_air`) |
| α | `alpha` | rad | geometric angle of attack (sign per `frames.md` §4) |
| Re | `re` | — | Reynolds number based on local chord |
| C_L, C_D | `C_L`, `C_D` | — | translational lift / drag coefficients |
| C_rot | `C_rot` | — | rotational-circulation coefficient |
| F_trans | `F_trans` | N | translational (circulatory + profile drag) force |
| F_rot | `F_rot` | N | rotational-circulation force |
| F_am | `F_am` | N | added-mass force |
| F_wing | `F_wing` | N | resultant wing force = Σ (F_trans + F_rot + F_am) |
| M_a | `M_aero` | N·m | aerodynamic moment |
| v_air | `v_air_W` | m/s | air (wind) velocity in W; still air = 0 |

## Actuation, solver, experiment

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| τ_act | `tau_act` | N·m | actuator (abdomen joint) torque |
| τ_lim | `tau_limit` | N·m | actuator torque saturation limit |
| α̇_lim | `rate_limit` | rad/s | actuator rate limit |
| h | `dt` | s | integrator timestep |
| n_wb | `steps_per_wingbeat` | — | solver resolution per wingbeat |
| rtol, atol | `rtol`, `atol` | — | solver relative / absolute tolerance |

## Test and analysis

| Symbol | Code name | Unit | Meaning |
|---|---|---|---|
| ⟨·⟩_wb | `wingbeat_mean(...)` | — | wingbeat (cycle) average, derived from time-resolved data |
| ε | `eps` | — | generic small tolerance; tests state their own values explicitly |
