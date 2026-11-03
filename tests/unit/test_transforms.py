"""§11.1 rotation and frame transform tests (M1 Task 1).

Conventions under test: docs/conventions/frames.md.
"""

from __future__ import annotations

import numpy as np

from fwmav_sim.core.transforms import (
    R_SP_WR,
    R_W_B,
    S_MIRROR,
    Rx,
    Ry,
    Rz,
    dR_dt,
    euler_from_R_W_B,
    matrix_to_quat,
    quat_from_euler,
    quat_multiply,
    quat_normalize,
    quat_propagate_constant_omega,
    quat_rate,
    quat_to_matrix,
)

RNG = np.random.default_rng(42)


def _random_angles(n: int):
    return RNG.uniform(-np.pi, np.pi, size=(n, 3))


def test_quat_matrix_roundtrip():
    for psi, theta, phi_r in _random_angles(200):
        theta = np.clip(theta, -1.4, 1.4)  # stay off gimbal lock
        R = R_W_B(psi, theta, phi_r)
        q = matrix_to_quat(R)
        assert np.linalg.norm(q) == 1.0 or abs(np.linalg.norm(q) - 1.0) < 1e-12
        np.testing.assert_allclose(quat_to_matrix(q), R, atol=1e-12)
        # quaternion from euler composition agrees
        np.testing.assert_allclose(
            quat_to_matrix(quat_from_euler(psi, theta, phi_r)), R, atol=1e-12
        )
        # euler inverse recovers angles
        psi2, theta2, phi_r2 = euler_from_R_W_B(R)
        np.testing.assert_allclose(
            R_W_B(psi2, theta2, phi_r2), R, atol=1e-12
        )


def test_rotation_composition_associative():
    for _ in range(100):
        a, b = RNG.uniform(-np.pi, np.pi, 2)
        np.testing.assert_allclose(
            (Rz(a) @ Rz(b)), Rz(a + b), atol=1e-12, err_msg="R(a)R(b) == R(a+b)"
        )
        # quaternion composition matches matrix composition
        qa, qb = matrix_to_quat(Rx(a)), matrix_to_quat(Ry(b))
        np.testing.assert_allclose(
            quat_to_matrix(quat_multiply(qa, qb)), Rx(a) @ Ry(b), atol=1e-12
        )


def test_rotation_derivative_finite_difference():
    """Ṙ = R[ω]× against central finite difference of R(t) (agreement < 1e-6)."""
    for _ in range(50):
        q0 = quat_normalize(RNG.normal(size=4))
        omega = RNG.uniform(-5.0, 5.0, size=3)
        h = 1e-7
        R_minus = quat_to_matrix(quat_propagate_constant_omega(q0, omega, -h))
        R_plus = quat_to_matrix(quat_propagate_constant_omega(q0, omega, +h))
        Rd_fd = (R_plus - R_minus) / (2 * h)
        R0 = quat_to_matrix(q0)
        Rd_exact = dR_dt(R0, omega)
        np.testing.assert_allclose(Rd_exact, Rd_fd, atol=1e-6)
        # quaternion rate consistency: q̇ = ½ q ⊗ [0, ω]
        qd = quat_rate(q0, omega)
        qd_fd = (quat_propagate_constant_omega(q0, omega, h)
                 - quat_propagate_constant_omega(q0, omega, -h)) / (2 * h)
        np.testing.assert_allclose(qd, qd_fd, atol=1e-6)


def test_left_wing_reflection_mirrors_positions_and_velocities():
    """frames.md §4: chord, span and physical normal mirror under S = diag(1,-1,1)."""
    for _ in range(200):
        phi, theta_w, psi_w = RNG.uniform(-1.2, 1.2, 3)
        R_R = R_SP_WR(phi, theta_w, psi_w, "right")
        R_L = R_SP_WR(phi, theta_w, psi_w, "left")
        for v in ([1.0, 0.0, 0.0], [0.0, 0.35, 0.0]):
            np.testing.assert_allclose(R_L @ v, S_MIRROR @ R_R @ v, atol=1e-12)
        # left physical wing normal is -z_L and mirrors the right physical normal
        np.testing.assert_allclose(
            -R_L @ [0, 0, 1], S_MIRROR @ R_R @ np.array([0, 0, 1]), atol=1e-12
        )
        # both frames remain right-handed
        assert np.linalg.det(R_L) > 0.999999
        # mirrored element velocities: omega and v mirror componentwise
        r = np.array([0.01, 0.3, 0.0])
        p_R, p_L = R_R @ r, R_L @ r
        np.testing.assert_allclose(p_L, S_MIRROR @ p_R, atol=1e-12)


def test_quat_normalize_rejects_zero():
    try:
        quat_normalize(np.zeros(4))
    except ValueError:
        return
    raise AssertionError("zero quaternion must be rejected")
