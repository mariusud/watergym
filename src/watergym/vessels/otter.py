"""Maritime Robotics Otter USV, a 2 m twin-pontoon catamaran with two thrusters.
Coefficients from Fossen's PythonVehicleSimulator otter.py via
research/11-vessel-parameters.md section 2. Pontoons are boxes, so the draft differs from
Fossen's hull-form coefficients; reverse thrust is treated as symmetric."""

from watergym.geometry import box_mesh, box_samples
from watergym.hydrostatics import WATER_DENSITY
from watergym.rigid_body import RigidBody
from watergym.vessel import Thruster, Vessel

MASS_KG = 80.0
PONTOON_OFFSET_M = 0.395
PONTOON_SIZE = (2.0, 0.25, 0.45)
KEEL_BELOW_CG_M = 0.44


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
    deck = box_mesh((0.1, 0.0, pontoon_z - 0.25), (1.0, 1.0, 0.1))
    mesh = box_mesh(port, PONTOON_SIZE) + box_mesh(starboard, PONTOON_SIZE) + deck
    draft_m = MASS_KG / (WATER_DENSITY * 2 * PONTOON_SIZE[0] * PONTOON_SIZE[1])
    return Vessel(
        name="otter",
        body=body,
        hull=hull,
        mesh=mesh,
        thrusters=[
            Thruster((-1.0, side * PONTOON_OFFSET_M, pontoon_z), (1.0, 0.0, 0.0), 119.7)
            for side in (-1, 1)
        ],
        initial_eta=(0.0, 0.0, draft_m - KEEL_BELOW_CG_M, 0.0, 0.0, 0.0),
    )
