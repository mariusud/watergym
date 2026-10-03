"""International Moth foiling dinghy, flying in the vertical plane (surge, heave, pitch).
The sailor's roll balance is not modelled. Numbers from research/11-vessel-parameters.md
section 1; wand geometry, gearing, sailor mass and inertia there are estimates.

Body origin is the CG of boat plus sailor, placed 0.25 m above the hull bottom.
"""

import math

import torch
from torch import Tensor

from watergym.foils import Foil, helmbold_lift_slope
from watergym.geometry import box_mesh, box_samples
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
    wings = box_mesh((0.0, 0.0, -0.05), (0.8, 2.25, 0.03))
    mast = box_mesh((0.3, 0.0, -2.0), (0.05, 0.05, 4.0))
    mesh = box_mesh(hull_center, hull_size) + wings + mast
    for foil in (main_foil, rudder_foil):
        mesh = mesh + box_mesh(foil.position, (foil.chord_m, foil.span_m, 0.015))
    for strut in struts:
        mesh = mesh + box_mesh(strut.position, (strut.chord_m, 0.015, strut.span_m))

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
