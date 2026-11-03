# Reduced Longitudinal Thorax–Abdomen Model — Specification

**Status: FROZEN (Milestone 0), written before any M2 code exists** (stage file
Task 4). Symbols per `notation.md`; frames and signs per `frames.md`.

Scope: the Level-1 fidelity model of brief §6.5 — longitudinal motion only,
two concentrated masses, cycle-averaged wing forces (assumption **A-007**).
Used for analytical regression tests and early controller experiments. The
full nonlinear articulated model (Level 2) is M3 and must recover this model
in its planar restriction.

## 1. Geometry and assumptions

- Planar motion in the W x–z plane (x forward/north, z down).
- **Thorax:** point mass `m_t` at the origin of B (the thorax reference
  point), plus a scalar pitch inertia `I_t` about the thorax CoM. The
  point-mass and the inertia are both located at the B origin.
- **Abdomen:** point mass `m_a` concentrated at distance `l_a` from the
  joint along the abdomen axis. Abdomen rotary inertia about its own CoM is
  neglected (point-mass assumption); if a distributed abdomen is later
  needed, it enters as `I_a` and is labelled a model extension.
- **Joint:** revolute abdomen pitch joint at `r_J = (−d, 0)` in B
  (`d ≥ 0`: the joint lies `d` behind the thorax reference point along the
  body x-axis; z-offset is a vehicle parameter, zero in the reference case).
- Wing forces enter **only** as a cycle-averaged resultant (A-007) derived
  from a time-resolved M1 run — never from an independent formula
  (brief §3.3).
- Suspended-string configurations are a separate constraint model (M2),
  never conflated with free flight (brief §6.3).

## 2. Definitions (brief §10 M0 — no ambiguity permitted)

| Item | Definition |
|---|---|
| Pitch zero reference | θ = 0 ⇔ B coincides with W (thorax x-axis horizontal, nose north). θ > 0 = nose up. |
| Abdomen angle zero reference | α_abd = 0 ⇔ abdomen axis aligned with the thorax x-axis (body straight). α_abd > 0 = abdomen tip deflects **downward** (rotation about +y_B). |
| Thrust direction convention | The cycle-averaged wing-force resultant `F_wing_avg = (F_x, F_z)` in B decomposes into thrust `T ≡ F_x` along **+x_B** (body forward) and lift `L ≡ −F_z` along −z_B (up). Applied at the thrust application point `r_T` in B (wing-root midpoint; longitudinal model: `r_T = (x_T, 0)`). |
| CoM reference point | Positions are measured from the origin of B (thorax reference point); the instantaneous CoM is computed per brief §6.2 from the actual configuration, never from a visualization marker. |

## 3. State vector

```text
x = [x, z, θ, α_abd, ẋ, ż, θ̇, α̇_abd] ∈ R⁸
q = [x, z, θ, α_abd]  (generalized coordinates)
```

- `x, z`: position of the thorax reference point in W (z positive **down**).
- `θ`: thorax pitch (§2).
- `α_abd`: abdomen pitch relative to the thorax (§2).

## 4. Kinematics of the two masses

Abdomen axis unit vector (abdomen extends *backward* from the joint):

```text
u_W = ( −cos(θ + α_abd),  sin(θ + α_abd) )
```

Positions in W:

```text
r_t = (x, z)
r_J = (x − d cosθ,        z + d sinθ)
r_a = (x − d cosθ − l_a cos(θ+α_abd),  z + d sinθ + l_a sin(θ+α_abd))
```

Velocities (ẋ_a, ż_a from direct differentiation):

```text
ẋ_a = ẋ + d sinθ · θ̇ + l_a sin(θ+α_abd) · (θ̇ + α̇_abd)
ż_a = ż + d cosθ · θ̇ + l_a cos(θ+α_abd) · (θ̇ + α̇_abd)
```

Instantaneous whole-body CoM (brief §6.2, two-mass case):

```text
r_CoM(q) = (m_t r_t + m_a r_a) / (m_t + m_a)
```

## 5. Nonlinear equations of motion (Euler–Lagrange)

Kinetic and potential energy (z down ⇒ gravitational potential
`V = −(m_t z + m_a z_a) g`, g = +9.81 m/s²):

```text
T = ½ m_t (ẋ² + ż²) + ½ I_t θ̇²
  + ½ m_a (ẋ_a² + ż_a²)
V = −(m_t z + m_a z_a) g
```

With `L = T − V` and non-conservative generalized forces `Q` (wing-force
resultant, actuator torque at the abdomen joint, joint damping):

```text
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = Qᵢ ,   i = 1..4
```

written in the standard manipulator form

```text
M(q) q̈ + C(q, q̇) q̇ + G(q) = Q_wing(q, F_wing_avg) + Q_act(τ_act) + Q_damp(q̇)
```

- `M(q)`: 4×4 symmetric positive-definite mass matrix, entries read off T
  (e.g. `M_xx = m_t + m_a`, `M_xθ = m_a [d sinθ + l_a sin(θ+α_abd)]`,
  `M_αα = m_a l_a²`, …). The explicit symbolic expansion is **generated and
  verified** by `derivations/reduced_longitudinal.py` (SymPy) in M2 — that
  script is the independent analytical reference (brief §15); no hand-typed
  matrix entry enters the code without matching it to `< 1e-10`.
- `Q_wing`: with `F_wing_avg` applied at `r_T(q)`, `Q_wing,i = F·∂r_T/∂qᵢ`.
- `Q_act`: actuator torque `τ_act` about the abdomen joint maps as
  `Q_act = [0, 0, 0, τ_act]ᵀ`. Because α_abd is the *relative* joint angle,
  virtual work gives Q_θ = 0 explicitly: the actuator does work
  `τ·δ(θ+α) − τ·δθ = τ·δα`; the reaction on the thorax manifests through the
  mass-matrix coupling, not an explicit forcing term (brief §6.1 item 3).
- `Q_damp = [0, 0, 0, −c_a α̇_abd]ᵀ` with joint damping `c_a` (JointSpec).

## 6. Hover trim and small-angle linearization

Trim condition (hover, wings supplying cycle-averaged force):

```text
q₀ = (x₀, z₀, 0, α₀),   q̇₀ = 0
F_x = T₀ (trim thrust),   F_z = −(m_t + m_a) g   (lift = weight; z down)
moment about the instantaneous CoM vanishes
```

Linearizing `M(q)q̈ + C(q,q̇)q̇ + G(q) = Q(q,u)` about `(q₀, 0, u₀)` with
`δx = [δq; δq̇]`, `δu = [δT, δL, δτ_act]` (control input perturbations):

```text
A_lin = ⎡      0₄            I₄      ⎤        B_lin = ⎡  0₄ₓ₃  ⎤
        ⎣ −M₀⁻¹ K_q    −M₀⁻¹ D_q̇  ⎦               ⎣ M₀⁻¹ B_Q ⎦

M₀ = M(q₀)
K_q = ∂(G − Q)/∂q |₀        (4×4 stiffness-like Jacobian)
D_q̇ = ∂(C(q,q̇)q̇ − Q_damp)/∂q̇ |₀  = −∂Q_damp/∂q̇   (C q̇ vanishes at q̇=0)
B_Q = ∂Q/∂u |₀              (4×3 input Jacobian)
```

Notes:

- At `q̇ = 0` the Coriolis/centrifugal term contributes only through `K_q`
  (its value is zero; its partials w.r.t. q vanish because C(q,0) = 0), so
  `D_q̇` reduces to joint damping. This is stated so M2 implements the
  linearization from the same SymPy reference rather than re-deriving it.
- The SymPy script asserts that the numerical Jacobian of the nonlinear EOM
  matches `A_lin` to machine precision at 100 random perturbation scales —
  this is the "linearized model vs numerical Jacobian" test of the action
  plan (M2 tests).

## 7. Limiting cases M2 must reproduce

1. **Rigid limit:** `α̇_abd = α̈_abd = 0` ⇒ single-body planar dynamics with
   total mass `m_t + m_a` and pitch inertia about the CoM from parallel axis.
2. **Passive pendulum:** thorax pinned at the joint configuration of the
   suspended constraint model ⇒ small-oscillation frequency
   `ω = sqrt(m_a g d_cm / I_pivot)` to `< 1e-6` relative (action plan M2
   tests), with `d_cm` the pivot-to-CoM distance of the swinging assembly.
3. **Free-body gravity moment about the instantaneous CoM ≡ 0**
   (brief §6.3, machine precision).
4. **Energy:** frictionless passive case (`c_a = 0`, no actuation, no aero)
   conserves total energy to solver tolerance.
