"""Shapes for the vessels: volume samples for buoyancy, triangle meshes for drawing.

Buoyancy only ever sees boxes cut into cells. The meshes are for looks: hulls lofted
through cross-sections, NACA foils, tubes and cambered sails. All coordinates are body
frame (x forward, y starboard, z down), metres.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

Vec3 = tuple[float, float, float]
WHITE = (1.0, 1.0, 1.0)


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
    """Triangle mesh: vertices [V, 3], triangles [T, 3], vertex colours [V, 3] in 0..1.

    Triangles wind counter-clockwise seen from outside. The viewer averages normals over
    shared vertices, so a surface shades smooth where it shares vertices and shows a
    crease where it repeats them.
    """

    vertices: Tensor
    triangles: Tensor
    colors: Tensor | None = None

    def __post_init__(self) -> None:
        self.vertices = self.vertices.float()
        if self.colors is None:
            self.colors = torch.ones_like(self.vertices)

    def __add__(self, other: "Mesh") -> "Mesh":
        return Mesh(
            torch.cat((self.vertices, other.vertices)),
            torch.cat((self.triangles, other.triangles + len(self.vertices))),
            torch.cat((self.colors, other.colors)),
        )

    def painted(self, color: Vec3) -> "Mesh":
        return Mesh(
            self.vertices, self.triangles, torch.tensor(color).expand(len(self.vertices), 3)
        )


def merge(meshes: list[Mesh]) -> Mesh:
    total = meshes[0]
    for mesh in meshes[1:]:
        total = total + mesh
    return total


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


def skin(rings: Tensor) -> Mesh:
    """A closed solid through rings [S, K, 3] of K points each, capped flat at both ends.

    Neighbouring rings are joined by quads, so the surface is smooth along and around.
    Repeat a point in every ring to crease it there (a deck edge, a trailing edge).
    """
    num_rings, k = rings.shape[:2]
    ring, step = torch.meshgrid(torch.arange(num_rings - 1), torch.arange(k), indexing="ij")
    a = ring * k + step
    b = ring * k + (step + 1) % k
    c, d = a + k, b + k
    sides = torch.cat((torch.stack((a, c, d), -1), torch.stack((a, d, b), -1))).reshape(-1, 3)
    solid = Mesh(rings.reshape(-1, 3), sides) + cap(rings[0], flip=False) + cap(rings[-1])
    return outward(solid)


def cap(ring: Tensor, flip: bool = True) -> Mesh:
    """A fan from the ring's centroid, with its own vertices so the rim creases."""
    k = len(ring)
    step = torch.arange(k)
    fan = torch.stack((step, (step + 1) % k, torch.full_like(step, k)), -1)
    if flip:
        fan = fan.flip(-1)
    return Mesh(torch.cat((ring, ring.mean(0, keepdim=True))), fan)


def outward(mesh: Mesh) -> Mesh:
    """Rewind a closed mesh so its triangles face out (positive enclosed volume)."""
    if signed_volume(mesh) >= 0:
        return mesh
    return Mesh(mesh.vertices, mesh.triangles.flip(-1), mesh.colors)


def signed_volume(mesh: Mesh) -> float:
    a, b, c = mesh.vertices[mesh.triangles].unbind(1)
    return (a * torch.linalg.cross(b, c)).sum().item() / 6


def loft(sections: list[tuple[float, Tensor]]) -> Mesh:
    """A hull skinned through cross-section outlines [K, 2] of (y, z), each at its x."""
    return skin(
        torch.stack([torch.cat((torch.full((len(yz), 1), x), yz), -1) for x, yz in sections])
    )


def hull_section(
    half_beam: float, deck_z: float, keel_z: float, fullness: float = 2.0, points: int = 13
) -> Tensor:
    """Outline [points + 2, 2] of a hull section: a flat deck over a superellipse bottom.

    fullness 2 is an ellipse, below 2 the bottom turns to a V, above 2 to a box.
    The deck corners are repeated so the sheer line shows as a crease.
    """
    angle = torch.linspace(0, math.pi, points)
    shape = 2 / fullness
    y = half_beam * angle.cos().sign() * angle.cos().abs() ** shape
    z = deck_z + (keel_z - deck_z) * angle.sin().abs() ** shape
    bottom = torch.stack((y, z), -1)
    return torch.cat((bottom, bottom[[-1, 0]]))


def rounded_rectangle(
    half_width: float, half_height: float, radius: float, center: tuple[float, float] = (0, 0)
) -> Tensor:
    """Outline [16, 2] of a rectangle whose corners are quarter circles of `radius`."""
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    points = []
    for i, (sy, sz) in enumerate(corners):
        for angle in torch.linspace(0, math.pi / 2, 4) + i * math.pi / 2:
            points.append(
                (
                    center[0] + sy * (half_width - radius) + radius * math.cos(angle),
                    center[1] + sz * (half_height - radius) + radius * math.sin(angle),
                )
            )
    return torch.tensor(points)


def rounded_box(center: Vec3, size: Vec3, radius: float) -> Mesh:
    """A box with its four edges along x rounded."""
    x, y, z = center
    outline = rounded_rectangle(size[1] / 2, size[2] / 2, radius, (y, z))
    return loft([(x - size[0] / 2, outline), (x + size[0] / 2, outline)])


def perpendicular_axes(axis: Tensor) -> tuple[Tensor, Tensor]:
    helper = torch.tensor([0.0, 0.0, 1.0]) if axis[2].abs() < 0.9 else torch.tensor([1.0, 0, 0])
    u = torch.nn.functional.normalize(torch.linalg.cross(axis, helper), dim=0)
    return u, torch.linalg.cross(axis, u)


def revolve(start: Vec3, end: Vec3, radii: Tensor, sides: int = 12) -> Mesh:
    """A body of revolution from `start` to `end` with radius radii[i] at equal steps."""
    start_t, end_t = torch.tensor(start, dtype=torch.float), torch.tensor(end, dtype=torch.float)
    axis = torch.nn.functional.normalize(end_t - start_t, dim=0)
    u, v = perpendicular_axes(axis)
    angle = torch.arange(sides) * 2 * math.pi / sides
    circle = angle.cos()[:, None] * u + angle.sin()[:, None] * v
    along = torch.linspace(0, 1, len(radii))[:, None, None]
    centers = start_t + along * (end_t - start_t)
    return skin(centers + radii[:, None, None] * circle)


def tube(
    start: Vec3, end: Vec3, radius: float, end_radius: float | None = None, sides: int = 12
) -> Mesh:
    """A straight tube or cone with flat ends, for masts, frames and thruster ducts."""
    return revolve(
        start, end, torch.tensor([radius, radius if end_radius is None else end_radius]), sides
    )


def pod(start: Vec3, end: Vec3, radius: float, sides: int = 12, stations: int = 9) -> Mesh:
    """A streamlined body with rounded nose and tail, for foil fuselages and thruster pods."""
    t = torch.linspace(0, 1, stations)
    return revolve(start, end, radius * (math.pi * t).sin().clamp(min=0) ** 0.5, sides)


def naca_section(thickness_ratio: float, points: int = 10) -> Tensor:
    """NACA 00xx outline [2 * points - 1, 2] of (chord fraction from the leading edge,
    thickness / chord), running from the trailing edge over the top and back underneath.
    The two trailing-edge points are separate so the edge stays sharp."""
    x = (1 - torch.linspace(0, math.pi, points).cos()) / 2
    half = (
        5
        * thickness_ratio
        * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    )
    upper = torch.stack((x, half), -1).flip(0)
    lower = torch.stack((x, -half), -1)[1:]
    return torch.cat((upper, lower))


def foil_mesh(
    chord: float,
    span: float,
    thickness_ratio: float,
    taper: float,
    position: Vec3,
    span_axis: Vec3 = (0.0, 1.0, 0.0),
    stations: int = 9,
) -> Mesh:
    """A NACA 00xx foil centred on `position`, chord along x, tapering linearly from the
    middle of the span to tip chord = taper * root chord at both ends. `chord` is the mean
    chord, so the area is chord * span. The quarter-chord line is straight."""
    root = 2 * chord / (1 + taper)
    section = naca_section(thickness_ratio)
    span_dir = torch.tensor(span_axis, dtype=torch.float)
    chord_dir = torch.tensor([1.0, 0.0, 0.0])
    thickness_dir = torch.linalg.cross(chord_dir, span_dir)
    rings = []
    for s in torch.linspace(-0.5, 0.5, stations):
        local_chord = root * (1 - (1 - taper) * 2 * s.abs())
        forward = root / 4 + local_chord * (0.25 - section[:, :1])
        rings.append(
            torch.tensor(position, dtype=torch.float)
            + s * span * span_dir
            + forward * chord_dir
            + local_chord * section[:, 1:] * thickness_dir
        )
    return skin(torch.stack(rings))


def sheet(grid: Tensor) -> Mesh:
    """A two-sided surface through a grid [R, C, 3] of points, each side shaded on its own."""
    rows, columns = grid.shape[:2]
    r, c = torch.meshgrid(torch.arange(rows - 1), torch.arange(columns - 1), indexing="ij")
    a = r * columns + c
    b, d = a + 1, a + columns
    front = torch.cat((torch.stack((a, d, b), -1), torch.stack((b, d, d + 1), -1))).reshape(-1, 3)
    points = grid.reshape(-1, 3)
    return Mesh(points, front) + Mesh(points, front.flip(-1))


def sail(
    tack: Vec3,
    head: Vec3,
    clew: Vec3,
    camber: float = 0.08,
    roach: float = 0.0,
    rows: int = 14,
    columns: int = 8,
) -> Mesh:
    """A cambered sail: the luff runs tack to head, the leech head to clew bowed aft by
    `roach`, and every horizontal section is a parabola of depth camber * chord bulging
    to starboard."""
    tack_t, head_t, clew_t = torch.tensor(tack), torch.tensor(head), torch.tensor(clew)
    luff_dir = torch.nn.functional.normalize(head_t - tack_t, dim=0)
    aft = clew_t - tack_t
    aft = torch.nn.functional.normalize(aft - (aft @ luff_dir) * luff_dir, dim=0)
    leeward = torch.linalg.cross(luff_dir, aft)
    leeward = leeward if leeward[1] > 0 else -leeward
    height = torch.linspace(0, 1, rows)[:, None, None]
    across = torch.linspace(0, 1, columns)[None, :, None]
    luff = tack_t + height * (head_t - tack_t)
    leech = clew_t + height * (head_t - clew_t) + roach * (math.pi * height).sin() * aft
    flat = luff + across * (leech - luff)
    chord = (leech - luff).norm(dim=-1, keepdim=True)
    return sheet(flat + 4 * camber * chord * across * (1 - across) * leeward)
