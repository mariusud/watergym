"""Boxes, the one shape every vessel here is built from: as volume samples for buoyancy and
as triangle meshes for drawing. All coordinates are body frame (x forward, y starboard,
z down), metres."""

from dataclasses import dataclass

import torch
from torch import Tensor

Vec3 = tuple[float, float, float]


@dataclass
class VolumeSamples:
    """A hull cut into vertical-sided cells: centres [P, 3], volumes [P], heights [P]."""

    centers: Tensor
    volumes: Tensor
    heights: Tensor

    def __add__(self, other: "VolumeSamples") -> "VolumeSamples":
        return VolumeSamples(
            torch.cat((self.centers, other.centers)),
            torch.cat((self.volumes, other.volumes)),
            torch.cat((self.heights, other.heights)),
        )

    def to(self, device: str | torch.device) -> "VolumeSamples":
        return VolumeSamples(
            self.centers.to(device), self.volumes.to(device), self.heights.to(device)
        )


@dataclass
class Mesh:
    """Triangle mesh: vertices [V, 3] float, triangles [T, 3] int."""

    vertices: Tensor
    triangles: Tensor

    def __add__(self, other: "Mesh") -> "Mesh":
        return Mesh(
            torch.cat((self.vertices, other.vertices)),
            torch.cat((self.triangles, other.triangles + len(self.vertices))),
        )


def box_samples(center: Vec3, size: Vec3, cells: tuple[int, int, int]) -> VolumeSamples:
    """Split a box of `size` (length, beam, height) into a regular grid of cells."""
    axes = [
        c - s / 2 + (torch.arange(n) + 0.5) * s / n
        for c, s, n in zip(center, size, cells, strict=True)
    ]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    cell_volume = size[0] * size[1] * size[2] / (cells[0] * cells[1] * cells[2])
    num = len(centers)
    return VolumeSamples(
        centers,
        torch.full((num,), cell_volume),
        torch.full((num,), size[2] / cells[2]),
    )


def box_mesh(center: Vec3, size: Vec3) -> Mesh:
    """Four vertices per face, so every face shades flat."""
    corners = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    )
    vertices = corners * torch.tensor(size) + torch.tensor(center)
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    face_vertices = vertices[torch.tensor(faces).reshape(-1)]
    first = torch.arange(0, 24, 4)[:, None]
    triangles = torch.cat((first + torch.tensor([0, 1, 2]), first + torch.tensor([0, 2, 3])))
    return Mesh(face_vertices, triangles)
