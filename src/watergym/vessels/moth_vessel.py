"""International Moth foiling dinghy, flying in the vertical plane (surge, heave, pitch).
The sailor's roll balance is not modelled. Wand geometry, gearing, sailor mass and inertia
are estimates.

Body origin is the CG of boat plus sailor, placed 0.25 m above the hull bottom.
"""

import math

import torch
from torch import Tensor

from watergym.foils import Foil, helmbold_lift_slope
from watergym.geometry import (
    Mesh,
    Vec3,
    box_samples,
    foil_mesh,
    hull_section,
    loft,
    merge,
    pod,
    sail,
    sheet,
    tube,
)
from watergym.hydrostatics import WATER_DENSITY
from watergym.rigid_body import Pose, RigidBody
from watergym.vessel import Thruster, Vessel
from watergym.waves import Sea, elevation

MASS_KG = 115.0
HULL_BOTTOM_Z = 0.25
STRUT_LENGTH_M = 1.0
STRUT_CHORD_M = 0.118
FOIL_Z = HULL_BOTTOM_Z + STRUT_LENGTH_M
MAIN_FOIL_SPAN_M, MAIN_FOIL_CHORD_M = 0.988, 0.095
RUDDER_FOIL_SPAN_M, RUDDER_FOIL_CHORD_M = 0.65, 0.12
RUDDER_X = -2.3
FLAP_RANGE = math.radians(6)
# Thin-airfoil flap effectiveness for a 35 % chord plain flap (Abbott and von Doenhoff).
FLAP_EFFECTIVENESS = 0.7

WAND_PIVOT = (1.5, 0.0, 0.1)
WAND_LENGTH_M = 1.0
WAND_NEUTRAL_ANGLE = math.radians(45)
WAND_GEARING = 0.133

HULL = (1.0, 1.0, 1.0)
SAIL = (0.86, 0.88, 0.9)
CARBON = (0.09, 0.09, 0.1)
TRAMPOLINE = (0.42, 0.44, 0.46)
FOIL = (0.09, 0.09, 0.1)


def flat_plate_added_mass(foil: Foil) -> float:
    """rho pi (c/2)^2 b: the water a plate drags along when it heaves."""
    return WATER_DENSITY * math.pi * (foil.chord_m / 2) ** 2 * foil.span_m


def moth(main_foil_incidence_deg: float = 4.0, rudder_incidence_deg: float = 0.0) -> Vessel:
    main_foil = Foil(
        "main_foil",
        position=(0.0, 0.0, FOIL_Z),
        span_m=MAIN_FOIL_SPAN_M,
        chord_m=MAIN_FOIL_CHORD_M,
        incidence=math.radians(main_foil_incidence_deg),
        lift_slope=helmbold_lift_slope(MAIN_FOIL_SPAN_M / MAIN_FOIL_CHORD_M),
        flap_effectiveness=FLAP_EFFECTIVENESS,
        max_flap=FLAP_RANGE,
    )
    rudder_foil = Foil(
        "rudder_foil",
        position=(RUDDER_X, 0.0, FOIL_Z),
        span_m=RUDDER_FOIL_SPAN_M,
        chord_m=RUDDER_FOIL_CHORD_M,
        incidence=math.radians(rudder_incidence_deg),
        lift_slope=helmbold_lift_slope(RUDDER_FOIL_SPAN_M / RUDDER_FOIL_CHORD_M),
    )
    strut_center_z = HULL_BOTTOM_Z + STRUT_LENGTH_M / 2
    struts = [
        Foil(
            name, (x, 0.0, strut_center_z), STRUT_LENGTH_M, STRUT_CHORD_M, span_axis=(0.0, 0.0, 1.0)
        )
        for name, x in (("main_strut", 0.0), ("rudder_strut", RUDDER_X))
    ]

    body = RigidBody.from_diagonals(
        MASS_KG,
        inertia=[81.0, 81.0, 81.0],
        added_mass=[
            0.0,
            0.0,
            flat_plate_added_mass(main_foil) + flat_plate_added_mass(rudder_foil),
            0.0,
            flat_plate_added_mass(rudder_foil) * RUDDER_X**2,
            0.0,
        ],
        linear_damping=[0.0] * 6,
        quadratic_damping=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    hull_size = (3.355, 0.30, 0.35)
    hull_center = (0.0, 0.0, HULL_BOTTOM_Z - hull_size[2] / 2)
    mesh = moth_mesh(hull_size, main_foil, rudder_foil, struts)

    ride_height_m = 0.6
    return Vessel(
        name="moth",
        body=body,
        hull=box_samples(hull_center, hull_size, cells=(8, 1, 3)),
        mesh=mesh,
        foils=[main_foil, rudder_foil, *struts],
        thrusters=[
            Thruster(position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), max_force_n=200.0)
        ],
        free_dofs=(True, False, True, False, True, False),
        initial_eta=(0.0, 0.0, -(HULL_BOTTOM_Z + ride_height_m), 0.0, 0.0, 0.0),
        initial_nu=(8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def moth_mesh(hull_size: Vec3, main_foil: Foil, rudder_foil: Foil, struts: list[Foil]) -> Mesh:
    """White hull and sail, carbon rig and foils."""
    length, beam, depth = hull_size
    deck_z = HULL_BOTTOM_Z - depth
    stern_x = -length / 2

    def section(u: float) -> tuple[float, torch.Tensor]:
        """u runs 0 at the transom to 1 at the plumb bow; the keel rises at both ends."""
        bow = max(u - 0.45, 0) / 0.55
        stern = max(0.45 - u, 0) / 0.45
        half_beam = beam / 2 * (1 - bow**2) * (1 - 0.4 * stern**2)
        keel_z = HULL_BOTTOM_Z - 0.12 * bow**2 - 0.07 * stern**2
        fullness = 2.4 - 1.0 * bow
        outline = hull_section(half_beam, deck_z - 0.04 * bow**2, keel_z, fullness)
        return stern_x + u * length, outline

    hull = loft([section(u) for u in torch.linspace(0, 1, 18).tolist()])

    wing_root, wing_tip = beam / 2 - 0.02, 1.125
    root_z, tip_z = deck_z - 0.02, deck_z - 0.2
    front = ((0.35, wing_root, root_z), (0.12, wing_tip, tip_z))
    rear = ((-0.55, wing_root, root_z), (-0.4, wing_tip, tip_z))
    wings = []
    for side in (1, -1):
        corners = [(x, side * y, z) for x, y, z in (*front, *rear)]
        wings += [
            tube(corners[0], corners[1], 0.02),
            tube(corners[2], corners[3], 0.02),
            tube(corners[1], corners[3], 0.02),
        ]
        tramp = torch.tensor([[corners[0], corners[1]], [corners[2], corners[3]]])
        wings.append(sheet(tramp + torch.tensor([0.0, 0.0, 0.01])).painted(TRAMPOLINE))

    mast_x = 0.3
    mast = tube((mast_x, 0.0, deck_z), (mast_x, 0.0, -5.0), 0.03, 0.018)
    boom = tube((mast_x, 0.0, deck_z - 0.25), (-1.95, 0.0, deck_z - 0.3), 0.022)
    rig = sail(
        (mast_x - 0.03, 0.0, deck_z - 0.12),
        (mast_x - 0.03, 0.0, -4.9),
        (-1.95, 0.0, deck_z - 0.32),
        camber=0.07,
        roach=0.35,
    ).painted(SAIL)

    gantry_end = (RUDDER_X, 0.0, deck_z + 0.02)
    gantry = [
        tube((stern_x + 0.15, side * 0.09, deck_z + 0.02), gantry_end, 0.018) for side in (1, -1)
    ]
    wand_tip = (
        WAND_PIVOT[0] + WAND_LENGTH_M * math.cos(WAND_NEUTRAL_ANGLE),
        0.0,
        WAND_PIVOT[2] + WAND_LENGTH_M * math.sin(WAND_NEUTRAL_ANGLE),
    )
    wand = tube(WAND_PIVOT, wand_tip, 0.008)

    main_strut, rudder_strut = struts
    rudder_strut_top = gantry_end[2]
    rudder_strut_length = FOIL_Z - rudder_strut_top
    appendages = [
        foil_mesh(main_foil.chord_m, main_foil.span_m, 0.12, 0.45, main_foil.position),
        foil_mesh(rudder_foil.chord_m, rudder_foil.span_m, 0.12, 0.5, rudder_foil.position),
        foil_mesh(main_strut.chord_m, main_strut.span_m, 0.1, 1.0, main_strut.position, (0, 0, 1)),
        foil_mesh(
            rudder_strut.chord_m,
            rudder_strut_length,
            0.1,
            1.0,
            (RUDDER_X, 0.0, rudder_strut_top + rudder_strut_length / 2),
            (0, 0, 1),
        ),
        pod((0.22, 0.0, FOIL_Z), (-0.3, 0.0, FOIL_Z), 0.022),
        pod((RUDDER_X + 0.12, 0.0, FOIL_Z), (RUDDER_X - 0.12, 0.0, FOIL_Z), 0.016),
    ]
    carbon = merge([mast, boom, *wings, *gantry, wand]).painted(CARBON)
    return merge([hull.painted(HULL), carbon, merge(appendages).painted(FOIL), rig])


def wand_flap(pivot_height_m: Tensor) -> Tensor:
    """Flap angle [rad] set by the mechanical wand.

    The wand hangs from the bow and trails on the water, so its angle below horizontal is
    asin(pivot height / wand length). Flying higher swings it down toward vertical, which
    raises the flap and sheds lift.
    """
    wand_angle = torch.asin((pivot_height_m / WAND_LENGTH_M).clamp(0, 1))
    return (WAND_GEARING * (WAND_NEUTRAL_ANGLE - wand_angle)).clamp(-FLAP_RANGE, FLAP_RANGE)


def wand_action(sea: Sea, t: Tensor, eta: Tensor, sail_command: float = 0.65) -> Tensor:
    """The full Moth action [sail, flap] with the flap driven by the wand."""
    pivot = Pose.from_eta(eta).to_world(torch.tensor([WAND_PIVOT], device=eta.device))
    pivot_height = -(pivot[..., 2] + elevation(sea, pivot, t))[:, 0]
    sail = torch.full_like(pivot_height, sail_command)
    return torch.stack((sail, wand_flap(pivot_height) / FLAP_RANGE), dim=-1)
