"""Buoyancy and Froude-Krylov forces from volume samples against the instantaneous sea.

Linear wave pressure satisfies -grad(p) = rho (a_water - g), so the pressure force on a
small submerged volume V is rho V (a_water - g). Summed over the wet part of the hull this
gives floating, restoring moments and Froude-Krylov wave excitation in one expression.
"""

import torch
from torch import Tensor

from watergym.geometry import VolumeSamples
from watergym.rigid_body import Pose
from watergym.waves import GRAVITY, Sea, elevation, orbital_acceleration

WATER_DENSITY = 1025.0


def submerged_fraction(depth: Tensor, height: Tensor | float) -> Tensor:
    """Fraction of a vertical extent `height` below the surface, given the depth of its centre."""
    return (depth / height + 0.5).clamp(0, 1)


def hydrostatic_force(
    samples: VolumeSamples,
    sea: Sea,
    t: Tensor,
    pose: Pose,
    density: float = WATER_DENSITY,
) -> Tensor:
    """Buoyancy plus Froude-Krylov force and moment about the CG in the body frame [envs, 6]."""
    points = pose.to_world(samples.centers)
    depth = points[..., 2] + elevation(sea, points, t)
    wet_volume = samples.volumes * submerged_fraction(depth, samples.heights)
    gravity = torch.tensor([0.0, 0.0, GRAVITY], device=points.device)
    force_world = density * wet_volume[..., None] * (orbital_acceleration(sea, points, t) - gravity)
    force_body = pose.to_body(force_world)
    moment_body = torch.linalg.cross(samples.centers.expand_as(force_body), force_body)
    return torch.cat((force_body.sum(1), moment_body.sum(1)), dim=-1)
