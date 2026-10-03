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
TRAMPOLINE = (0.2, 0.21, 0.23)
LINE = (0.3, 0.3, 0.32)
# Foils are drawn mid grey: black carbon vanishes under the translucent sea.
FOIL = (0.55, 0.56, 0.58)


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
    """A Mach2-style Moth: narrow lofted hull with a crowned deck, swept wing racks with
    trampolines, a 5.1 m mast with a square-top deck-sweeper sail, boom and vang, the
    rudder gantry and tiller, the bow wand, and NACA foils on NACA struts placed exactly
    where the physics puts them."""
    length, beam, depth = hull_size
    deck_z = HULL_BOTTOM_Z - depth
    stern_x = -length / 2

    def section(u: float) -> tuple[float, Tensor]:
        """u runs 0 at the transom to 1 at the plumb bow; V forward, U aft, rocker at
        both ends."""
        bow = max(u - 0.4, 0) / 0.6
        stern = max(0.4 - u, 0) / 0.4
        half_beam = beam / 2 * (1 - bow**1.3) * (1 - 0.35 * stern**2)
        keel_z = HULL_BOTTOM_Z - 0.13 * bow**2 - 0.05 * stern**2
        sheer_z = deck_z - 0.05 * bow**2
        outline = hull_section(half_beam, sheer_z, keel_z, 2.4 - 1.1 * bow, deck_camber=0.03)
        return stern_x + u * length, outline

    hull = loft([section(u) for u in torch.linspace(0, 1, 26).tolist()])

    rack_z, tip_z = deck_z - 0.01, deck_z - 0.17
    wings, tramps, shrouds = [], [], []
    mast_x, hounds_z = 0.3, -3.6
    for side in (1, -1):
        front_root, front_tip = (0.4, side * 0.13, rack_z), (-0.3, side * 1.12, tip_z)
        rear_root, rear_tip = (-1.0, side * 0.12, rack_z), (-0.9, side * 1.12, tip_z)
        wings += [
            tube(front_root, front_tip, 0.018),
            tube(rear_root, rear_tip, 0.018),
            tube(front_tip, rear_tip, 0.022),
        ]
        tramp = torch.tensor([[front_root, front_tip], [rear_root, rear_tip]])
        tramps.append(sheet(tramp + torch.tensor([0.0, 0.0, 0.012])))
        chainplate = (0.05, side * 0.45, (rack_z + tip_z) / 2 - 0.02)
        shrouds.append(tube((mast_x, 0.0, hounds_z), chainplate, 0.004))

    mast = tube((mast_x, 0.0, deck_z - 0.02), (mast_x, 0.0, deck_z - 5.1), 0.03, 0.017)
    boom_z, clew_x = deck_z - 0.24, -1.85
    boom = tube((mast_x - 0.03, 0.0, boom_z), (clew_x - 0.05, 0.0, boom_z - 0.02), 0.022)
    vang = tube((mast_x - 0.02, 0.0, deck_z - 0.06), (mast_x - 0.55, 0.0, boom_z), 0.01)
    rig = sail(
        (mast_x - 0.03, 0.0, deck_z - 0.1),
        (mast_x - 0.03, 0.0, deck_z - 5.05),
        (clew_x, 0.0, boom_z - 0.03),
        camber=0.08,
        roach=0.3,
        head_width=0.38,
    ).painted(SAIL)

    rudder_head = (RUDDER_X, 0.0, deck_z + 0.03)
    gantry = [
        tube((stern_x + 0.06, side * 0.085, deck_z + 0.02), rudder_head, 0.016) for side in (1, -1)
    ]
    tiller_end = (-1.15, 0.0, deck_z - 0.07)
    tiller = tube(rudder_head, tiller_end, 0.012)
    extension = tube(tiller_end, (-0.75, 0.85, tip_z - 0.25), 0.007)
    wand_tip = (
        WAND_PIVOT[0] + WAND_LENGTH_M * math.cos(WAND_NEUTRAL_ANGLE),
        0.0,
        WAND_PIVOT[2] + WAND_LENGTH_M * math.sin(WAND_NEUTRAL_ANGLE),
    )
    wand = tube(WAND_PIVOT, wand_tip, 0.007)

    main_strut, rudder_strut = struts
    rudder_strut_length = FOIL_Z - rudder_head[2]
    vertical = (0.0, 0.0, 1.0)
    foils = [
        foil_mesh(main_foil.chord_m, main_foil.span_m, 0.11, 0.4, main_foil.position),
        foil_mesh(rudder_foil.chord_m, rudder_foil.span_m, 0.11, 0.45, rudder_foil.position),
        foil_mesh(main_strut.chord_m, main_strut.span_m, 0.12, 1.0, main_strut.position, vertical),
        foil_mesh(
            rudder_strut.chord_m,
            rudder_strut_length,
            0.12,
            1.0,
            (RUDDER_X, 0.0, rudder_head[2] + rudder_strut_length / 2),
            vertical,
        ),
        pod((0.2, 0.0, FOIL_Z), (-0.32, 0.0, FOIL_Z), 0.022),
        pod((RUDDER_X + 0.1, 0.0, FOIL_Z), (RUDDER_X - 0.12, 0.0, FOIL_Z), 0.016),
    ]
    return merge(
        [
            hull.painted(HULL),
            merge([mast, boom, vang, *wings, *gantry, tiller, wand]).painted(CARBON),
            merge([extension, *shrouds]).painted(LINE),
            merge(tramps).painted(TRAMPOLINE),
            merge(foils).painted(FOIL),
            rig,
        ]
    )


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
