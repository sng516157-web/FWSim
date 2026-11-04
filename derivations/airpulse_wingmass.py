"""Derive the AirPulse wing-mass variant vehicle (post-M5 item 1, DR-006).

Gu et al. 2026 (arXiv:2602.06811) state the wings are 38% of the total mass,
but Table A1 gives WHOLE-robot mass properties at neutral stroke. Splitting
the wings out as separate mass bodies (schema ``WingSpec.mass``) therefore
requires subtracting the planform-model wing inertia at the neutral
configuration from the Table A1 body inertia — otherwise the wing mass is
double-counted.

Neutral configuration: stroke/pitch/deviation zero, so R_WR = I (right) /
Rx(pi) (left); the stroke plane carries the fixed 50 deg V-dihedral (DR-005).
The body frame B origin sits at the neutral-stroke CG (airpulse.yaml), so the
wing inertia is taken about that point; by left/right symmetry the remaining
body CoM stays at the origin.

FINDING (recorded in docs/milestone_reports/P1_wing_mass.md): the paper's
"wing assemblies ~38% of flight-ready mass" is INCONSISTENT with the Table A1
whole-robot inertia under a uniform-plate wing model — 38% at uniform
density produces more inertia about the roll axis at neutral than the
measured whole-robot value (the assembly mass must be concentrated at the
root mechanism, not in the flapping membrane). The script computes the
largest per-wing uniform-plate mass that keeps the body-inertia remainder
positive definite (the consistency bound), and the variant vehicle uses a
membrane mass estimate below it (provenance: estimated, reasoning in the
YAML). Body mass/inertia = Table A1 minus the neutral-config wing model, so
total mass and neutral inertia remain exactly the paper's values.

Run: python derivations/airpulse_wingmass.py  (prints the YAML blocks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fwmav_sim.core.transforms import Rx
from fwmav_sim.dynamics.wing_inertia import wing_mass_properties
from fwmav_sim.experiment.config import load_vehicle
from fwmav_sim.vehicle.schema import Parameter, Provenance

ROOT = Path(__file__).resolve().parent.parent

WING_MASS_FRACTION = 0.38  # Gu et al. 2026: wing ASSEMBLIES ~38% of total mass
MEMBRANE_MASS_ESTIMATE = 0.5e-3  # kg per wing (estimated; see module docstring)


def _neutral_wing_inertia(vehicle, m_wing: float) -> np.ndarray:
    """Both wings' inertia about the robot CG at the neutral configuration."""
    I_wings = np.zeros((3, 3))
    for w in vehicle.wings:
        props = wing_mass_properties(
            w.model_copy(update={"mass": Parameter(
                value=m_wing, unit="kg",
                provenance=Provenance(source="estimated"))})
        )
        # neutral stroke: R_WR = I (right) / Rx(pi) (left); dihedral only
        gamma = w.dihedral_angle.value
        sigma = -1.0 if w.side == "right" else 1.0
        R_B_SP = Rx(sigma * gamma)
        R_WR = np.eye(3) if w.side == "right" else Rx(np.pi)
        R_W_wing = R_B_SP @ R_WR
        hinge = np.asarray(w.hinge_location.value, dtype=float)
        r_c = hinge + R_W_wing @ props.r_com_WR  # about B origin (= robot CG)
        I_c = R_W_wing @ props.I_com_WR @ R_W_wing.T
        I_wings += I_c + m_wing * (float(r_c @ r_c) * np.eye(3) - np.outer(r_c, r_c))
    return I_wings


def main() -> None:
    vehicle = load_vehicle(ROOT / "reference_vehicles" / "airpulse.yaml")
    body = vehicle.bodies[0]
    m_total = body.mass.value
    I_table = np.asarray(body.inertia.value, dtype=float)

    # consistency bound: the wing inertia scales linearly with wing mass
    I_unit = _neutral_wing_inertia(vehicle, 1.0)
    from scipy.optimize import brentq
    m_bound = brentq(lambda m: float(np.linalg.eigvalsh(I_table - m * I_unit).min()),
                     1e-5, m_total / 2, xtol=1e-9)
    m_assembly = WING_MASS_FRACTION * m_total / 2
    print(f"consistency bound: m_wing <= {m_bound * 1e3:.2f} g per wing")
    print(f"  (38% assemblies = {m_assembly * 1e3:.2f} g per wing -> remainder "
          f"{'NOT ' if m_bound < m_assembly else ''}positive definite: "
          "assembly mass must be root-concentrated)")

    m_wing = MEMBRANE_MASS_ESTIMATE
    if m_wing >= m_bound:
        raise SystemExit("membrane estimate exceeds the consistency bound")
    m_body = m_total - 2 * m_wing
    I_wings = m_wing * I_unit
    I_body = I_table - I_wings
    eig = np.linalg.eigvalsh(I_body)
    print(f"\nm_wing (each)  = {m_wing * 1e3:.3f} g")
    print(f"m_body         = {m_body * 1e3:.3f} g")
    print("wing inertia about robot CG (neutral):\n", I_wings)
    print("body inertia remainder:\n", repr(I_body.tolist()))
    print("remainder eigenvalues:", eig, "(must stay positive)")
    if eig.min() <= 0:
        raise SystemExit("ERROR: inertia remainder not positive definite")


if __name__ == "__main__":
    main()
