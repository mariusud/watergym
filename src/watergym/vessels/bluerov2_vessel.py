"""BlueROV2 Heavy, a small open-frame ROV. Mass, volume, added mass and damping from
von Benzon et al. 2022 (JMSE 10:1898, Table A1). The eight-thruster layout is
simplified to four vectored horizontal thrusters and four vertical ones."""

import math

from watergym.geometry import VolumeSamples, box_mesh, box_samples
from watergym.rigid_body import RigidBody
from watergym.vessel import Thruster, Vessel

FRAME_SIZE = (0.46, 0.58, 0.38)
DISPLACED_VOLUME_M3 = 0.0134


def bluerov2() -> Vessel:
    body = RigidBody.from_diagonals(
        13.5,
        inertia=[0.26, 0.23, 0.37],
        added_mass=[6.36, 7.12, 18.68, 0.189, 0.135, 0.222],
        linear_damping=[13.7, 0.0, 33.0, 0.0, 0.8, 0.0],
        quadratic_damping=[141.0, 217.0, 190.0, 1.19, 0.47, 1.5],
    )
    frame = box_samples((0.0, 0.0, -0.01), FRAME_SIZE, cells=(4, 4, 4))
    hull = VolumeSamples(
        frame.centers,
        frame.volumes * DISPLACED_VOLUME_M3 / frame.volumes.sum(),
        frame.heights,
    )
    diagonal = 1 / math.sqrt(2)
    horizontal = [
        Thruster((x, y, 0.0), (diagonal, -math.copysign(diagonal, x * y), 0.0), 40.0)
        for x in (0.15, -0.15)
        for y in (0.11, -0.11)
    ]
    vertical = [
        Thruster((x, y, 0.0), (0.0, 0.0, 1.0), 30.0) for x in (0.12, -0.12) for y in (0.22, -0.22)
    ]
    return Vessel(
        name="bluerov2",
        body=body,
        hull=hull,
        mesh=box_mesh((0.0, 0.0, 0.0), FRAME_SIZE),
        thrusters=horizontal + vertical,
        density=1000.0,
        initial_eta=(0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    )
