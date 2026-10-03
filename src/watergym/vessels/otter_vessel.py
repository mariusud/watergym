"""Maritime Robotics Otter USV, a 2 m twin-pontoon catamaran with two thrusters.
Coefficients from Fossen's PythonVehicleSimulator otter.py.
Pontoons are boxes, so the draft differs from
Fossen's hull-form coefficients; reverse thrust is treated as symmetric."""

import math

import torch
from torch import Tensor

from watergym.geometry import (
    Mesh,
    box_samples,
    foil_mesh,
    hull_section,
    loft,
    merge,
    pod,
    ring,
    rounded_box,
    tube,
)
from watergym.hydrostatics import WATER_DENSITY
from watergym.rigid_body import RigidBody
from watergym.vessel import Thruster, Vessel

MASS_KG = 80.0
PONTOON_OFFSET_M = 0.395
PONTOON_SIZE = (2.0, 0.25, 0.45)
KEEL_BELOW_CG_M = 0.44

HULL = (1.0, 1.0, 1.0)
DECK = (0.3, 0.32, 0.35)
HATCH = (0.62, 0.64, 0.67)
DARK = (0.1, 0.1, 0.11)


def otter() -> Vessel:
    body = RigidBody.from_diagonals(
        MASS_KG,
        inertia=[16.68, 21.52, 15.10],
        added_mass=[5.5, 82.5, 55.0, 3.34, 17.21, 25.67],
        linear_damping=[77.55, 162.5, 605.4, 62.07, 266.6, 42.65],
        quadratic_damping=[0.0, 200.9, 0.0, 0.0, 0.0, 426.5],
    )
    pontoon_z = KEEL_BELOW_CG_M - PONTOON_SIZE[2] / 2
    port = (0.0, -PONTOON_OFFSET_M, pontoon_z)
    starboard = (0.0, PONTOON_OFFSET_M, pontoon_z)
    cells = (10, 2, 9)
    hull = box_samples(port, PONTOON_SIZE, cells) + box_samples(starboard, PONTOON_SIZE, cells)
    draft_m = MASS_KG / (WATER_DENSITY * 2 * PONTOON_SIZE[0] * PONTOON_SIZE[1])
    return Vessel(
        name="otter",
        body=body,
        hull=hull,
        mesh=otter_mesh(),
        thrusters=[
            Thruster((-1.0, side * PONTOON_OFFSET_M, pontoon_z), (1.0, 0.0, 0.0), 119.7)
            for side in (-1, 1)
        ],
        initial_eta=(0.0, 0.0, draft_m - KEEL_BELOW_CG_M, 0.0, 0.0, 0.0),
    )


def otter_mesh() -> Mesh:
    """Two round-bilged pontoons with raked bows, a bridge deck with a hatch, a GNSS mast,
    and under each stern a thruster pod with a three-blade propeller in a guard ring."""
    length, beam, depth = PONTOON_SIZE
    keel_z = KEEL_BELOW_CG_M
    deck_z = keel_z - depth

    def section(u: float, side: float) -> tuple[float, Tensor]:
        """u runs 0 at the transom to 1 at the stem; the last third narrows and lifts."""
        bow = max(u - 0.62, 0) / 0.38
        stern = max(0.1 - u, 0) / 0.1
        half_beam = beam / 2 * (1 - bow**1.6) * (1 - 0.1 * stern)
        keel = keel_z - 0.3 * bow**1.5 - 0.04 * stern
        outline = hull_section(half_beam, deck_z, keel, fullness=2.8, deck_camber=0.02)
        return -length / 2 + u * length, outline + torch.tensor([side * PONTOON_OFFSET_M, 0.0])

    stations = torch.linspace(0, 1, 24).tolist()
    pontoons = [loft([section(u, side) for u in stations]) for side in (-1, 1)]

    deck_width = 2 * PONTOON_OFFSET_M + 0.1
    deck = rounded_box((0.05, 0.0, deck_z - 0.04), (1.2, deck_width, 0.1), 0.04)
    hatch = rounded_box((0.1, 0.0, deck_z - 0.1), (0.55, 0.42, 0.03), 0.012)
    mast_x = -0.3
    mast = tube((mast_x, 0.0, deck_z - 0.09), (mast_x, 0.0, deck_z - 0.6), 0.016)
    crossbar = tube((mast_x, -0.12, deck_z - 0.59), (mast_x, 0.12, deck_z - 0.59), 0.012)
    pucks = [
        pod((mast_x, side * 0.12, deck_z - 0.58), (mast_x, side * 0.12, deck_z - 0.64), 0.055)
        for side in (-1, 1)
    ]

    drive = []
    for side in (-1, 1):
        y = side * PONTOON_OFFSET_M
        x, z = -length / 2 + 0.15, keel_z + 0.12
        prop = (x - 0.14, y, z)
        drive += [
            pod((x + 0.14, y, z), (x - 0.16, y, z), 0.04),
            foil_mesh(0.14, 0.14, 0.15, 1.0, (x + 0.02, y, keel_z + 0.05), (0.0, 0.0, 1.0)),
            ring(prop, (1.0, 0.0, 0.0), 0.085, 0.06, 0.008),
        ]
        for k in range(3):
            angle = 2 * math.pi * k / 3
            blade = (0.0, math.cos(angle), math.sin(angle))
            center = (prop[0], y + 0.045 * blade[1], z + 0.045 * blade[2])
            drive.append(foil_mesh(0.035, 0.07, 0.12, 0.6, center, blade, stations=3))
    return merge(
        [
            merge(pontoons).painted(HULL),
            deck.painted(DECK),
            hatch.painted(HATCH),
            merge([mast, crossbar]).painted(DARK),
            merge(pucks).painted(HULL),
            merge(drive).painted(DARK),
        ]
    )
