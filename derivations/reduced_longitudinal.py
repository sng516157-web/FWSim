"""SymPy derivation of the reduced longitudinal thorax-abdomen model (M2 Task 3).

Dev-only (SymPy is a dev dependency). Implements docs/conventions/
reduced_model_spec.md via Euler-Lagrange and generates the pure-NumPy
reference module src/fwmav_sim/dynamics/_reduced_reference.py. Rerunning
regenerates the file and asserts the content is unchanged (diff check), so the
committed reference and the derivation can never drift apart.

Models derived:
  1. free  — q = [x, z, theta, alpha_abd], the M2/M3 trust anchor
  2. pend  — suspended rigid-pivot model, q_s = [phi_p, alpha_abd]
     (constraints/suspended.py; string modelled as a rigid massless pivot,
     valid for the small-oscillation swing mode — see M2 report)

Planar rotations use the frozen frames.md convention: Ry(theta) maps
(bx, bz) -> (c bx + s bz, -s bx + c bz) in the (x, z-down) plane.

Run: python derivations/reduced_longitudinal.py
"""

from __future__ import annotations

import pathlib
import sys

import sympy as sp

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "fwmav_sim" / "dynamics" / "_reduced_reference.py"

# ---------------------------------------------------------------- free model
x, z, th, al = sp.symbols("x z theta alpha_abd", real=True)
xd, zd, thd, ald = sp.symbols("xd zd theta_d alpha_abd_d", real=True)
m_t, m_a, I_t, I_a, d, l_a, g = sp.symbols("m_t m_a I_t I_a d l_a g", positive=True)

sth, cth = sp.sin(th), sp.cos(th)
stha, ctha = sp.sin(th + al), sp.cos(th + al)

# thorax CoM at (x, z); abdomen CoM per reduced_model_spec.md section 4
xa = x - d * cth - l_a * ctha
za = z + d * sth + l_a * stha

xa_d = sp.diff(xa, x) * xd + sp.diff(xa, z) * zd + sp.diff(xa, th) * thd + sp.diff(xa, al) * ald
za_d = sp.diff(za, x) * xd + sp.diff(za, z) * zd + sp.diff(za, th) * thd + sp.diff(za, al) * ald

T_free = (sp.Rational(1, 2) * m_t * (xd**2 + zd**2) + sp.Rational(1, 2) * I_t * thd**2
          + sp.Rational(1, 2) * I_a * (thd + ald)**2
          + sp.Rational(1, 2) * m_a * (xa_d**2 + za_d**2))
V_free = -(m_t * z + m_a * za) * g  # z down, g = +9.81 (frames.md section 1.1)

q = sp.Matrix([x, z, th, al])
qd = sp.Matrix([xd, zd, thd, ald])
FREE_PARAMS = (m_t, m_a, I_t, d, l_a, g)


def derive_free():
    n = 4
    M = sp.Matrix(n, n, lambda i, j: sp.expand(sp.diff(T_free, qd[i], qd[j])))
    bias_vel = sp.Matrix.zeros(n, 1)
    for i in range(n):
        bias_vel[i] = sp.simplify(
            sum(sp.diff(T_free, qd[i], q[j]) * qd[j] for j in range(n))
            - sp.diff(T_free, q[i])
        )
    Gv = sp.Matrix([sp.simplify(sp.diff(V_free, q[i])) for i in range(n)])
    return M, bias_vel, Gv


# ------------------------------------------------------- pendulum (suspended)
phi = sp.symbols("phi_p", real=True)
phid = sp.symbols("phi_p_d", real=True)
utx, utz, jx, jz = sp.symbols("u_tx u_tz j_x j_z", real=True)  # thorax CoM / joint
                                                                # offsets from pivot, in B
P_x, P_z = sp.symbols("P_x P_z", real=True)

qp = sp.Matrix([phi, al])
qpd = sp.Matrix([phid, ald])
PEND_PARAMS = (m_t, m_a, I_t, utx, utz, jx, jz, l_a, g)

sph, cph = sp.sin(phi), sp.cos(phi)
# Ry(phi) @ (ux, uz) = (c ux + s uz, -s ux + c uz)   [planar, frames.md]
rt_x = P_x + cph * utx + sph * utz
rt_z = P_z - sph * utx + cph * utz
rj_x = P_x + cph * jx + sph * jz
rj_z = P_z - sph * jx + cph * jz
# abdomen CoM: joint + Ry(phi + alpha) @ (-l_a, 0)
spa, cpa = sp.sin(phi + al), sp.cos(phi + al)
ra_x = rj_x - l_a * cpa
ra_z = rj_z + l_a * spa


def _diff2(expr, q, qd):
    return sum(sp.diff(expr, q[i]) * qd[i] for i in range(len(q)))


rt_xd, rt_zd = _diff2(rt_x, qp, qpd), _diff2(rt_z, qp, qpd)
ra_xd, ra_zd = _diff2(ra_x, qp, qpd), _diff2(ra_z, qp, qpd)

T_pend = (sp.Rational(1, 2) * m_t * (rt_xd**2 + rt_zd**2) + sp.Rational(1, 2) * I_t * phid**2
          + sp.Rational(1, 2) * I_a * (phid + ald)**2
          + sp.Rational(1, 2) * m_a * (ra_xd**2 + ra_zd**2))
V_pend = -(m_t * rt_z + m_a * ra_z) * g


def derive_pend():
    n = 2
    M = sp.Matrix(n, n, lambda i, j: sp.expand(sp.diff(T_pend, qpd[i], qpd[j])))
    bias_vel = sp.Matrix.zeros(n, 1)
    for i in range(n):
        bias_vel[i] = sp.simplify(
            sum(sp.diff(T_pend, qpd[i], qp[j]) * qpd[j] for j in range(n))
            - sp.diff(T_pend, qp[i])
        )
    Gv = sp.Matrix([sp.simplify(sp.diff(V_pend, qp[i])) for i in range(n)])
    return M, bias_vel, Gv


# ------------------------------------------------------------------ generate

def _fmt_matrix(name, M, args, doc):
    n = M.rows
    entries = ", ".join(sp.pycode(sp.simplify(M[i, j])) for i in range(n) for j in range(n))
    return (
        f"def {name}({args}):\n"
        f'    """{doc}"""\n'
        f"    return np.array([{entries}], dtype=float).reshape({n}, {n})\n"
    )


def _fmt_vector(name, v, args, doc):
    entries = ", ".join(sp.pycode(sp.simplify(v[i])) for i in range(v.rows))
    return (
        f"def {name}({args}):\n"
        f'    """{doc}"""\n'
        f"    return np.array([{entries}], dtype=float)\n"
    )


#: Hand-written (template) NumPy assembly of the 8x8 linearization from the
#: symbolically generated blocks. Inserted verbatim into the generated module.
_A_LIN_ASSEMBLY = '''
def free_A_lin(theta_0, alpha_abd_0, b_x, b_z, F_x, F_z, tau_act,
               m_t, m_a, I_t, I_a, d, l_a, g):
    """Linearized state matrix A (8x8) of the free reduced model about
    (q0 = [x, z, theta_0, alpha_abd_0], qd = 0) with a point force (F_x, F_z)
    in the BODY frame B applied at (b_x, b_z) in B and actuator torque tau_act
    on alpha_abd.

    A = [[0, I], [M0^-1 (dQ/dq - dG/dq) - M0^-1 dM/dq_k M0^-1 h, 0]]
    (bias is quadratic in qd, so the velocity blocks vanish at qd = 0;
    damping, when configured, is added by the runtime model).
    """
    M0 = free_M(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a)
    M0inv = np.linalg.inv(M0)
    h = free_Q_point(theta_0, b_x, b_z, F_x, F_z) + np.array([0.0, 0.0, 0.0, tau_act]) \\
        - free_G(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a, g)
    dG = free_A_lin_dG_dq(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a, g)
    dQ = free_A_lin_dQ_dq(theta_0, b_x, b_z, F_x, F_z)
    qdd0 = M0inv @ h
    dM = [free_A_lin_dM_dtheta(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a),
          free_A_lin_dM_dalpha(theta_0, alpha_abd_0, m_t, m_a, I_t, I_a, d, l_a)]
    A = np.zeros((8, 8))
    A[:4, 4:] = np.eye(4)
    bottom = M0inv @ (dQ - dG)
    for k, dM in ((2, dM[0]), (3, dM[1])):
        bottom[:, k] += -M0inv @ dM @ qdd0
    A[4:, :4] = bottom
    return A
'''


def main() -> int:
    M_f, bias_f, G_f = derive_free()
    M_p, bias_p, G_p = derive_pend()

    # point-force Jacobian: force (Fx, Fz) in the BODY frame B applied at
    # (bx, bz) in B (thrust rotates with the body — reduced_model_spec.md §2)
    bx, bz, Fx_b, Fz_b = sp.symbols("b_x b_z F_x_B F_z_B", real=True)
    rp_x = x + cth * bx + sth * bz
    rp_z = z - sth * bx + cth * bz
    J = sp.Matrix(4, 2, lambda i, j: sp.diff([rp_x, rp_z][j], q[i]))  # rows: q, cols: r
    Fx_w = cth * Fx_b + sth * Fz_b   # Ry(theta) @ F_B
    Fz_w = -sth * Fx_b + cth * Fz_b
    Q_point = J @ sp.Matrix([Fx_w, Fz_w])

    # Linearization blocks about (q0, qd = 0), force F0 at (b_x, b_z), torque tau
    # on alpha (Q_alpha += tau; Q_theta unchanged — alpha_abd is the relative
    # joint angle, reduced_model_spec.md §5). qdd = M^-1 h, h = Q - bias - G.
    # At qd = 0: bias = 0, d(bias)/dq = 0, d(bias)/dqd = 0 (quadratic in qd), so
    #   d(qdd)/dq_k = M0^-1 (dQ/dq_k - dG/dq_k) - M0^-1 (dM/dq_k) M0^-1 h
    #   d(qdd)/dqd  = 0   (damping, when configured, is added by the runtime model)
    # Blocks are generated symbolically; assembly (numeric M^-1) happens in the
    # generated module — no symbolic matrix inverse needed.
    dG_dq = sp.Matrix(4, 4, lambda i, j: sp.simplify(sp.diff(G_f[i], q[j])))
    dQ_dq = sp.Matrix(4, 4, lambda i, j: sp.simplify(sp.diff(Q_point[i], q[j])))
    dM_dth = sp.Matrix(4, 4, lambda i, j: sp.simplify(sp.diff(M_f[i, j], th)))
    dM_dal = sp.Matrix(4, 4, lambda i, j: sp.simplify(sp.diff(M_f[i, j], al)))

    header = (
        '"""GENERATED FILE - do not edit by hand.\n\n'
        "Regenerate with: python derivations/reduced_longitudinal.py\n"
        "Source: docs/conventions/reduced_model_spec.md (Euler-Lagrange, SymPy).\n"
        '"""\n\nfrom __future__ import annotations\n\nimport math\n\nimport numpy as np\n\n\n'
    )
    free_args = "theta, alpha_abd, m_t, m_a, I_t, I_a, d, l_a"
    free_args_g = free_args + ", g"
    free_vel_args = "theta, alpha_abd, theta_d, alpha_abd_d, m_t, m_a, d, l_a"
    parts = [header]
    parts.append(_fmt_matrix("free_M", M_f, free_args,
                             "Mass matrix M(q) of the free reduced model;"
                             " q = [x, z, theta, alpha_abd]."))
    parts.append("\n")
    parts.append(_fmt_vector("free_bias", bias_f, free_vel_args,
                             "Velocity-product (Coriolis/centrifugal) vector of the free model."))
    parts.append("\n")
    parts.append(_fmt_vector("free_G", G_f, free_args_g,
                             "Gravity vector G(q) = dV/dq of the free model."))
    parts.append("\n")
    parts.append(_fmt_vector("free_Q_point", Q_point,
                             "theta, b_x, b_z, F_x_B, F_z_B",
                             "Generalized force of a point force (F_x, F_z) in the BODY frame B "
                             "applied at (b_x, b_z) in B."))
    parts.append("\n")
    pend_args = "phi_p, alpha_abd, m_t, m_a, I_t, I_a, u_tx, u_tz, j_x, j_z, l_a"
    parts.append(_fmt_matrix("pend_M", M_p, pend_args,
                             "Mass matrix of the suspended-pendulum model;"
                             " q_s = [phi_p, alpha_abd]."))
    parts.append("\n")
    parts.append(_fmt_vector("pend_bias", bias_p,
                             ("phi_p, alpha_abd, phi_p_d, alpha_abd_d, m_t, m_a, "
                              "u_tx, u_tz, j_x, j_z, l_a"),
                             "Velocity-product vector of the pendulum model."))
    parts.append("\n")
    parts.append(_fmt_vector("pend_G", G_p, pend_args + ", g",
                             "Gravity vector of the pendulum model."))
    parts.append("\n")
    parts.append(_fmt_matrix("free_A_lin_dG_dq", dG_dq, free_args_g,
                             "dG/dq block for the free-model linearization."))
    parts.append("\n")
    parts.append(_fmt_matrix("free_A_lin_dQ_dq", dQ_dq, "theta, b_x, b_z, F_x_B, F_z_B",
                             "dQ/dq block (body-frame point force) for the free-model "
                             "linearization."))
    parts.append("\n")
    parts.append(_fmt_matrix("free_A_lin_dM_dtheta", dM_dth, free_args,
                             "dM/dtheta block for the free-model linearization."))
    parts.append("\n")
    parts.append(_fmt_matrix("free_A_lin_dM_dalpha", dM_dal, free_args,
                             "dM/dalpha block for the free-model linearization."))
    parts.append("\n")
    parts.append(_A_LIN_ASSEMBLY)

    content = "".join(parts) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") == content:
        print(f"reference unchanged: {OUT}")
    else:
        OUT.write_text(content, encoding="utf-8")
        print(f"reference regenerated: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
