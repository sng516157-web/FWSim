# Coordinate Frames, Sign Conventions and Transform Chains

**Status: FROZEN (Milestone 0).** Later milestones must not redefine anything in
this document. Changes require a decision record in `docs/decisions/` and a
revision of this file in the same commit.

Contract: `FWMAV_Simulator_Project_Brief.md` §10 (M0 acceptance): a second
developer can determine every frame, state and force direction from this
document without reading implementation code.

All quantities are SI. Angles are radians. All rotation matrices below are
**direction-cosine matrices**: `R_A_B` maps the coordinates of a vector
expressed in frame `B` to coordinates expressed in frame `A`, where `A` is the
**parent** of `B`. The columns of `R_A_B` are the `B` frame's axes expressed in
`A`. Composition reads parent ← child: `v_W = R_W_B R_B_SP R_SP_WR R_WR_BE v_BE`.

Elementary rotations (right-handed, positive = right-hand rule about the axis):

```text
        ⎡1   0    0 ⎤          ⎡ cα  0   sα⎤          ⎡cα  −sα  0⎤
Rx(α) = ⎢0  cα  −sα⎥   Ry(α) = ⎢ 0  1   0 ⎥   Rz(α) = ⎢sα   cα  0⎥
        ⎣0  sα   cα⎦          ⎣−sα  0   cα⎦          ⎣ 0    0  1⎦
```

with `cα = cos α`, `sα = sin α`.

---

## 1. The six frames

Machine-readable mirror: `src/fwmav_sim/core/frames.py` (kept in sync by
`tests/unit/test_frames_registry.py`).

| Frame | Name | Parent | Definition |
|---|---|---|---|
| W | world | — | NED: x north, y east, z **down**. Inertial. Gravity `g = [0, 0, +9.81] m/s²`. |
| B | body (thorax) | W | x forward, y right, z down. Origin at the thorax reference point (see §7). Coincides with W at zero attitude. |
| SP | stroke plane | B | B rotated by the stroke-plane angle β about the B y-axis: `R_B_SP = Ry(β)`. |
| WR_R | wing root, right | SP | Span axis along +y (outboard along the right wing), chord along x, per the Sane & Dickinson convention. |
| WR_L | wing root, left | SP | Right-handed mirror of WR_R; see §4. Span along −y_SP, chord along +x. |
| BE | blade element | WR_R or WR_L | Origin at the element centre; axes aligned with the parent WR frame. |

### 1.1 World frame W (NED)

```text
        x_N (north)
        ↑
        |
        |
        o------→ y_E (east)
       /
      ↙
    z_D (down, into the page at 45° here)

gravity: g = [0, 0, +9.81] m/s²  (points along +z_W, i.e. DOWN)
```

An aircraft in level unaccelerated flight therefore experiences **specific
force along −z_B**, and lift in hover points along **−z_W**. Positive z is
*down* everywhere in this project.

### 1.2 Body frame B

x forward (toward the head), y right (starboard), z down (ventral). At zero
attitude B coincides with W. The origin of B is the **thorax reference point**,
defined per vehicle in its `VehicleDefinition` (for the reference vehicle: the
thorax geometric centre; see `docs/conventions/reduced_model_spec.md` §2).

### 1.3 Attitude: ZYX Euler + quaternion

Attitude of B relative to W is parametrized two ways:

**Euler angles (ZYX, intrinsic):** yaw ψ → pitch θ → roll φ_r, applied as

```text
R_W_B = Rz(ψ) · Ry(θ) · Rx(φ_r)
```

- ψ (yaw): rotation about +z_W. ψ = 0 points the nose north (+x_W).
- θ (pitch): rotation about the intermediate y-axis. **θ > 0 = nose up**
  (body x-axis gains a −z_W component, and +z_W is down).
- φ_r (roll): rotation about the body x-axis. **φ_r > 0 = right wing down**
  (body y-axis gains a +z component).

Note the naming: `φ_r` (roll) is distinct from the wing flapping angle `φ`
(§3). In code, roll is always `phi_r`, never `phi`.

**Quaternion:** scalar-first `q = (w, x, y, z)`, Hamilton convention,
representing the same rotation as `R_W_B`, stored normalized
(`w² + x² + y² + z² = 1`). Renormalization after every accepted integrator
step is mandatory (brief §5.7). The corresponding DCM:

```text
R_W_B = ⎡1−2(y²+z²)   2(xy−wz)   2(xz+wy)⎤
        ⎢2(xy+wz)   1−2(x²+z²)   2(yz−wx)⎥
        ⎣2(xz−wy)   2(yz+wx)   1−2(x²+y²)⎦
```

Body angular velocity `ω_B = [p, q, r]ᵀ` (components in B; p roll rate,
q pitch rate, r yaw rate) maps to quaternion rate via

```text
q̇ = ½ q ⊗ [0, p, q, r]ᵀ        (Hamilton product)
```

### 1.4 Stroke-plane frame SP

```text
R_B_SP = Ry(β)
```

**Positive β:** right-hand rule about +y_B. Since Ry(β) maps
x̂ → [cos β, 0, −sin β], a positive β raises the anterior (+x) end of the
stroke plane (and lowers the posterior end), because +z_B is down:

```text
side view (looking along −y_B, x forward, z down):

   β = 0                     β > 0
                              x_SP
   x_SP                       ↗
   ───────→  wingbeat        ───────↗  stroke plane tilted anterior-up
   ═══════   plane          ╱
                          ╱  β (rotation about +y_B, into the page)
   z ↓                     z ↓
```

β = 0: stroke plane contains the body x–y plane (horizontal at zero attitude).

**Per-wing dihedral γ (added in DR-005, M4-prep):** an optional per-wing roll
tilt of the stroke plane about the body x-axis, applied before β:

```text
R_B_SP_wing = Rx(σ γ) · Ry(β),   σ = −1 (right), +1 (left)
```

so γ > 0 raises both wingtips (V-dihedral, e.g. the AirPulse platform's 50°).
γ = 0 recovers the frozen chain exactly. Mirroring is consistent:
S Rx(−γ) S = Rx(+γ).

## 2. Wing-root frame WR (right wing)

With all wing angles zero, WR_R coincides with SP: chord along +x_SP
(leading edge forward), span along +y_SP (outboard), +z normal to the stroke
plane pointing "below" it (down at β = 0).

## 3. Wing angles (Sane & Dickinson convention)

Three prescribed angles orient the wing relative to SP, applied in the order
**stroke → deviation → pitch**:

```text
R_SP_WR_R = Rz(−φ) · Rx(ψ_w) · Ry(θ_w)
```

Time-domain definitions for prescribed sinusoidal kinematics (M1):

```text
φ(t)   = φ_amp · cos(2π f t)                      stroke (flapping)
ψ_w(t) = ψ_amp · sin(2π f t + δ_ψ)                deviation (optional, default 0)
θ_w(t) = θ_off + θ_amp · sin(2π f t + δ_θ)        wing pitching (rotation)
```

with f the flapping frequency in Hz. Fourier-extended profiles
(`Σ aₙ sin(2π n f t + δ_n)`) use the same zero references.

- **φ — stroke (flapping) angle.** Measured in the stroke plane from the
  +y_SP (lateral) axis; **positive φ moves the wing tip forward (+x_SP)**.
  φ = 0 at mid-stroke (wing pointing laterally). With
  `φ(t) = φ_amp cos(2π f t)`, t = 0 is the forward extreme.
  The `Rz(−φ)` form: a rotation about +z_SP by `−φ` maps the span axis
  ŷ → [sin φ, cos φ, 0].
- **ψ_w — deviation angle.** Rotation about the chordwise x-axis of the
  flapped frame. **Positive ψ_w moves the wing tip toward +z_SP** (below the
  stroke plane at β = 0). ψ_w = 0: the wing beats in the stroke plane.
- **θ_w — wing pitch (rotation) angle.** Rotation about the spanwise y-axis
  of the deviated frame. θ_w = 0: the wing chord lies in the (flapped,
  deviated) stroke plane. **Positive θ_w rotates the leading edge toward
  −z** (up at β = 0) — right-hand rule about the +y_WR span axis.
  Equivalently, positive θ_w *increases* the geometric angle of attack of a
  wing moving in the +x_WR direction.

```text
top view of SP (z down into page)          side view of deviated frame
y_SP                                       (looking along +y_WR span):
 ↑        wing at φ > 0                      −z (up)
 ↑      ↙  tip forward                        ↑
 ↑    ↙                                       ↑   θ_w > 0: leading edge up
 ↑  ↙  span                                   o──────→ x (chord, forward)
 o───────→ x_SP                               chord rotates toward −z
 φ measured from +y toward +x                wing pitch about +y (into page)
```

**Rotation-rate sign convention for θ_w:** the wing pitching rate
`θ̇_w > 0` is a nose-up rotation of the wing about its span axis.

## 4. Left wing: the reflection operator S

Define the reflection

```text
S = diag(1, −1, 1)        (reflection across the x–z plane, det S = −1)
```

S is applied at the **base of the WR chain** (rightmost factor), i.e. it
mirrors the wing-root base orientation. The frozen left-wing transform is

```text
R_SP_WR_L = Rz(+φ) · Rx(−ψ_w) · Ry(θ_w) · Rx(π)
```

where `Rx(π) = diag(1, −1, −1)` is the right-handed left-wing base orientation
(chord forward, span along −y_SP, z-axis pointing *up*, opposite to the
physical wing normal). Numerically verified identity (M0, random
configurations): with `R_SP_WR_R` from §3 and `R_SP_WR_L` above,

- chord: `R_L x̂ = S R_R x̂`
- span: `R_L ŷ = S R_R ŷ`
- physical wing normal: `−R_L ẑ = S R_R ẑ`

**Sign rules.**

| Quantity | Right wing | Left wing |
|---|---|---|
| commanded φ(t), θ_w(t), ψ_w(t) | as defined | **same functions** (symmetric flapping = same commands) |
| flap rotation | Rz(−φ) | Rz(+φ) |
| deviation rotation | Rx(+ψ_w) | Rx(−ψ_w) |
| pitch rotation | Ry(+θ_w) | Ry(+θ_w) |
| base orientation | I | Rx(π) |
| frame handedness | right | right (z_L anti-parallel to physical normal) |
| physical wing normal | +ẑ_BE | **−ẑ_BE** |
| geometric angle of attack α | measured from chord toward **+ẑ_BE** | measured from chord toward **−ẑ_BE** |
| element position/velocity/acceleration | p, v, a | S·p, S·v, S·a |
| aerodynamic force (in SP) | F | **S·F** (x and z unchanged, y negated) |
| aerodynamic moment (in SP) | M | **S′ M** with S′ = diag(−1, 1, −1): M_x and M_z negated, M_y unchanged |

The moment rule follows from M = r × F with mirrored r and F.

## 5. Blade-element frame BE

Each rigid wing (assumptions A-001, A-005) is discretized spanwise into N
blade elements. BE frame i: origin at the element centre (span station s_i,
measured along +y_WR from the wing root), axes aligned with the parent WR:

```text
R_WR_BE = I          (axes parallel; BE differs from WR only by translation)
p_BE/WR = [0, s_i, 0]ᵀ  in WR coordinates
```

## 6. Full transform chains and transport equations

Rotation chain (right wing; left wing per §4):

```text
v_W = R_W_B · R_B_SP · R_SP_WR · R_WR_BE · v_BE
    = Rz(ψ)Ry(θ)Rx(φ_r) · Ry(β) · Rz(−φ)Rx(ψ_w)Ry(θ_w) · I · v_BE
```

**Position** of a point P fixed in a child frame, expressed in the world frame:

```text
p_P/W = p_B/W + R_W_B ( p_SP/B + R_B_SP ( p_WR/SP + R_SP_WR p_P/WR ) )
```

For the reference vehicle the wing root sits at the body-frame hinge location
`p_WR/B` given in `WingSpec.hinge_location`, i.e. `p_SP/B + R_B_SP p_WR/SP`
is replaced by the configured `p_WR/B`.

**Velocity transport** (rigid chain, angular velocity ω of child w.r.t.
parent, components in the child frame; ∘ denotes the cross product):

```text
v_P = v_O + R ( ω × r_P )        P fixed in the rotating frame
```

with the standard addition of relative (flapping) motion: for the wing
kinematics chain the element velocity in SP is

```text
v_BE/SP = ω_WR/SP × r_BE/WR ,   ω_WR/SP from (φ̇, ψ̇_w, θ̇_w) mapped through the chain
```

and the velocity relative to the air (brief §5.4) is

```text
v_rel = v_element_in_W − v_air_in_W
```

**Acceleration transport:**

```text
a_P = a_O + R ( ω̇ × r_P + ω × (ω × r_P) )      (+ 2 ω × v_rel for moving points)
```

These are the equations M1 implements and finite-difference tests check.

## 7. Reduced longitudinal model reference points

State, mass and force conventions for the thorax–abdomen model are frozen in
`docs/conventions/reduced_model_spec.md`. Cross-references fixed there:
pitch zero reference = θ = 0 (B coincides with W); abdomen angle zero
reference = abdomen aligned with the thorax x-axis; thrust direction = +x_B;
CoM reference point = origin of B.

## 8. Common mistakes (review checklist)

1. **Gravity sign.** g = [0, 0, **+9.81**] because z is *down*. Lift in hover
   is −z. A simulation that "floats up" with positive z acceleration has the
   sign wrong.
2. **Quaternion order.** Scalar-first `(w, x, y, z)`, Hamilton. Never
   `(x, y, z, w)`. Renormalize after every step.
3. **Roll vs flapping.** Roll is `phi_r`; wing stroke is `phi`. Same Greek
   letter, different quantities — the code names differ.
4. **β direction.** Positive β raises the *front* of the stroke plane
   (§1.4). It is easy to assume the opposite from helicopter conventions.
5. **Left-wing mirroring.** Do not mirror the final rotation matrix
   (S·R is improper, det = −1). Use the frozen chain of §4, and remember the
   left physical normal is −ẑ_BE. Symmetric commands give symmetric forces
   F_L = S·F_R — M1 tests this exactly.
6. **φ zero reference.** φ is measured from the *lateral* (+y_SP) axis, not
   from x_SP. `φ(t) = φ_amp cos(2πft)` starts at the forward extreme.
7. **θ_w zero reference.** θ_w = 0 puts the chord *in* the stroke plane; it
   is not measured from the relative wind.
8. **Rotation order.** ZYX for attitude (Rz Ry Rx); stroke → deviation →
   pitch for the wing (Rz(−φ) Rx(ψ_w) Ry(θ_w)). Reordering either chain is a
   different convention, not a different parametrization.
9. **DCM direction.** `R_A_B` maps B-coords → A-coords. The chain reads
   parent ← child. `R_B_A = R_A_Bᵀ`.
10. **Units.** SI everywhere internally; angles in radians. Convert only at
    dataset import boundaries (M4), with the conversion recorded.
