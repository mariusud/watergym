"""BlueROV2 Heavy, a small open-frame ROV. Mass, volume, added mass and damping from
von Benzon et al. 2022 (JMSE 10:1898, Table A1). The eight-thruster layout is
simplified to four vectored horizontal thrusters and four vertical ones."""

import math

import torch

from watergym.geometry import Mesh, VolumeSamples, box_samples, merge, pod, ring, rounded_box, tube
from watergym.rigid_body import RigidBody
from watergym.vessel import Thruster, Vessel

FRAME_SIZE = (0.46, 0.58, 0.38)
DISPLACED_VOLUME_M3 = 0.0134

FRAME = (0.08, 0.08, 0.09)
FOAM = (0.07, 0.07, 0.08)
FAIRING = (0.1, 0.33, 0.75)
ACRYLIC = (0.78, 0.84, 0.9)
ALUMINIUM = (0.6, 0.62, 0.66)
THRUSTER = (0.16, 0.16, 0.18)


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
        mesh=bluerov2_mesh(horizontal + vertical),
        thrusters=horizontal + vertical,
        density=1000.0,
        initial_eta=(0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    )


def bluerov2_mesh(thrusters: list[Thruster]) -> Mesh:
    """The Heavy at its real 457 x 575 x 253 mm: a tube frame, black foam and the blue
    fairing on top, the electronics tube with its dome and the battery tube underneath,
    and a ducted thruster at every physics thruster position and direction, with guard
    rings around the four vertical ones."""
    half_x, half_y, half_z = 0.457 / 2 - 0.015, 0.575 / 2 - 0.015, 0.253 / 2 - 0.012
    frame = []
    for y in (-half_y, half_y):
        corners = [(-half_x, y, -half_z), (half_x, y, -half_z), (half_x, y, half_z)]
        corners.append((-half_x, y, half_z))
        frame += [tube(corners[i], corners[(i + 1) % 4], 0.012) for i in range(4)]
        frame.append(tube((0.0, y, -half_z), (0.0, y, half_z), 0.01))
    for x in (-half_x, half_x):
        frame += [tube((x, -half_y, z), (x, half_y, z), 0.012) for z in (-half_z, half_z)]

    top = -half_z - 0.01
    foam = [
        rounded_box((0.0, side * 0.1, top + 0.035), (0.4, 0.07, 0.07), 0.015) for side in (-1, 1)
    ]
    fairing = rounded_box((0.0, 0.0, top + 0.03), (0.42, 0.13, 0.06), 0.03)
    electronics = tube((-0.15, 0.0, -0.01), (0.15, 0.0, -0.01), 0.057)
    flanges = [tube((x, 0.0, -0.01), (x + 0.025, 0.0, -0.01), 0.064) for x in (-0.175, 0.15)]
    dome = pod((0.1, 0.0, -0.01), (0.24, 0.0, -0.01), 0.054)
    battery = tube((-0.13, 0.0, 0.085), (0.13, 0.0, 0.085), 0.035)

    ducts, guards = [], []
    for thruster in thrusters:
        center, axis = torch.tensor(thruster.position), torch.tensor(thruster.direction)
        hub_ends = [tuple((center + offset * axis).tolist()) for offset in (-0.06, 0.05)]
        ducts += [ring(thruster.position, thruster.direction, 0.044, 0.07, 0.007)]
        ducts += [pod(*hub_ends, 0.022)]
        if thruster.direction[2]:
            guards.append(ring(thruster.position, thruster.direction, 0.062, 0.03, 0.008))
    return merge(
        [
            merge(frame + guards).painted(FRAME),
            merge(foam).painted(FOAM),
            fairing.painted(FAIRING),
            merge([electronics, dome]).painted(ACRYLIC),
            merge([*flanges, battery]).painted(ALUMINIUM),
            merge(ducts).painted(THRUSTER),
        ]
    )
