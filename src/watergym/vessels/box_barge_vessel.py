"""A 10 x 4 x 2 m box barge of uniform density floating at 1 m draft: the hydrostatics
reference."""

from watergym.geometry import Mesh, box_mesh, box_samples, merge, tube
from watergym.hydrostatics import WATER_DENSITY
from watergym.rigid_body import RigidBody
from watergym.vessel import Vessel
from watergym.waves import GRAVITY

LENGTH_M, BEAM_M, HEIGHT_M, DRAFT_M = 10.0, 4.0, 2.0, 1.0
MASS_KG = WATER_DENSITY * LENGTH_M * BEAM_M * DRAFT_M

HULL = (0.55, 0.58, 0.62)
TRIM = (0.9, 0.9, 0.88)
FENDER = (0.12, 0.12, 0.13)


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
        mesh=box_barge_mesh(),
        initial_eta=(0.0, 0.0, HEIGHT_M / 2 - DRAFT_M, 0.0, 0.0, 0.0),
    )


def box_barge_mesh() -> Mesh:
    """The box itself, with a low bulwark around the deck, a rubbing strake along the
    sheer and a bollard at each corner."""
    deck_z = -HEIGHT_M / 2
    hull = box_mesh((0.0, 0.0, 0.0), (LENGTH_M, BEAM_M, HEIGHT_M))
    wall, rise = 0.08, 0.25
    bulwark = [
        box_mesh((side * (LENGTH_M - wall) / 2, 0.0, deck_z - rise / 2), (wall, BEAM_M, rise))
        for side in (-1, 1)
    ] + [
        box_mesh((0.0, side * (BEAM_M - wall) / 2, deck_z - rise / 2), (LENGTH_M, wall, rise))
        for side in (-1, 1)
    ]
    strake = [
        box_mesh((0.0, side * BEAM_M / 2, deck_z + 0.15), (LENGTH_M + 0.12, 0.12, 0.14))
        for side in (-1, 1)
    ] + [
        box_mesh((side * LENGTH_M / 2, 0.0, deck_z + 0.15), (0.12, BEAM_M + 0.12, 0.14))
        for side in (-1, 1)
    ]
    bollards = [
        tube((x, y, deck_z), (x, y, deck_z - 0.35), 0.1)
        for x in (-LENGTH_M / 2 + 0.6, LENGTH_M / 2 - 0.6)
        for y in (-BEAM_M / 2 + 0.4, BEAM_M / 2 - 0.4)
    ]
    return merge(
        [
            hull.painted(HULL),
            merge(bulwark + bollards).painted(TRIM),
            merge(strake).painted(FENDER),
        ]
    )
