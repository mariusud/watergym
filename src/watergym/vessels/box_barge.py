"""A 10 x 4 x 2 m box barge of uniform density floating at 1 m draft: the hydrostatics
reference. Numbers from research/11-vessel-parameters.md section 4."""

from watergym.geometry import box_mesh, box_samples
from watergym.hydrostatics import WATER_DENSITY
from watergym.rigid_body import RigidBody
from watergym.vessel import Vessel
from watergym.waves import GRAVITY

LENGTH_M, BEAM_M, HEIGHT_M, DRAFT_M = 10.0, 4.0, 2.0, 1.0
MASS_KG = WATER_DENSITY * LENGTH_M * BEAM_M * DRAFT_M


def box_barge(heave_damping_ratio: float = 0.1) -> Vessel:
    inertia = [
        MASS_KG / 12 * (BEAM_M**2 + HEIGHT_M**2),
        MASS_KG / 12 * (LENGTH_M**2 + HEIGHT_M**2),
        MASS_KG / 12 * (LENGTH_M**2 + BEAM_M**2),
    ]
    added_mass = [
        0.05 * MASS_KG,
        MASS_KG,
        MASS_KG,
        0.3 * inertia[0],
        0.3 * inertia[1],
        0.3 * inertia[2],
    ]
    heave_stiffness = WATER_DENSITY * GRAVITY * LENGTH_M * BEAM_M
    heave_damping = heave_damping_ratio * 2 * ((MASS_KG + added_mass[2]) * heave_stiffness) ** 0.5
    body = RigidBody.from_diagonals(
        MASS_KG,
        inertia,
        added_mass,
        linear_damping=[0.0, 0.0, heave_damping, 5e4, 5e5, 0.0],
        quadratic_damping=[
            0.5 * WATER_DENSITY * BEAM_M * DRAFT_M,
            0.5 * WATER_DENSITY * LENGTH_M * DRAFT_M,
            0.0,
            0.0,
            0.0,
            1e5,
        ],
    )
    size = (LENGTH_M, BEAM_M, HEIGHT_M)
    return Vessel(
        name="box_barge",
        body=body,
        hull=box_samples((0.0, 0.0, 0.0), size, cells=(10, 16, 10)),
        mesh=box_mesh((0.0, 0.0, 0.0), size),
        initial_eta=(0.0, 0.0, HEIGHT_M / 2 - DRAFT_M, 0.0, 0.0, 0.0),
    )
