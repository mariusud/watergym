"""Draw WaterGym tensors with Newton's viewer (install the `viz` extra).

Each environment gets a cell of a grid. The cell follows its vessel: the sea patch is
sampled around the vessel's (x, y), so a fast boat stays in view while the water streams past.

Newton draws in a z-up frame, WaterGym computes in NED (z down). FLIP = diag(1, -1, -1)
maps one to the other, for points (F p) and for rotations (F R F).
"""

import math

import numpy as np
import torch
import warp as wp
from newton import viewer as newton_viewer
from torch import Tensor

from watergym.foils import FoilLoads
from watergym.rigid_body import Pose
from watergym.vessel import Vessel
from watergym.waves import Sea, elevation

FLIP = torch.tensor([1.0, -1.0, -1.0])
SEA_COLOR = (0.15, 0.45, 0.65)
HULL_COLOR = (0.92, 0.9, 0.85)
FORCE_COLOR = (1.0, 0.45, 0.1)
# Roughness, metallic, checker, texture on: the texture carries the mesh's part colours.
VESSEL_MATERIAL = (0.5, 0.0, 0.0, 1.0)


def make_viewer(kind: str = "gl", headless: bool = False, width: int = 1280, height: int = 720):
    """`gl` opens a window (or renders offscreen with headless=True), `viser` serves a browser
    view (needs `pip install viser`), `usd` records to watergym.usd (needs `usd-core`),
    `null` draws nothing."""
    if kind == "gl":
        return newton_viewer.ViewerGL(width=width, height=height, headless=headless)
    if kind == "viser":
        return newton_viewer.ViewerViser()
    if kind == "usd":
        return newton_viewer.ViewerUSD(output_path="watergym.usd", num_frames=None)
    if kind == "null":
        return newton_viewer.ViewerNull(num_frames=10**9)
    raise ValueError(f"unknown viewer {kind!r}")


def to_viewer_frame(points_ned: Tensor) -> Tensor:
    return points_ned * FLIP.to(points_ned.device)


def quaternion_xyzw(rotation: Tensor) -> Tensor:
    """Unit quaternions [N, 4] (x, y, z, w) from rotation matrices [N, 3, 3]."""
    m = rotation
    w = 0.5 * torch.sqrt((1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]).clamp(min=1e-12))
    x = 0.5 * torch.sqrt((1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp(min=0)).copysign(
        m[:, 2, 1] - m[:, 1, 2]
    )
    y = 0.5 * torch.sqrt((1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2]).clamp(min=0)).copysign(
        m[:, 0, 2] - m[:, 2, 0]
    )
    z = 0.5 * torch.sqrt((1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]).clamp(min=0)).copysign(
        m[:, 1, 0] - m[:, 0, 1]
    )
    return torch.nn.functional.normalize(torch.stack((x, y, z, w), dim=-1), dim=-1)


def palette_texture(colors: Tensor) -> tuple[np.ndarray, Tensor]:
    """One texel per distinct vertex colour [1, N, 3] and each vertex's uv [V, 2] at the
    centre of its texel. A part's uv is constant, so the texture is never blended."""
    palette, slot = torch.unique(colors, dim=0, return_inverse=True)
    uvs = torch.stack(
        ((slot + 0.5) / len(palette), torch.full_like(slot, 0.5, dtype=torch.float)), -1
    )
    texels = (palette[None] * 255).round().to(torch.uint8).numpy()
    return texels, uvs.float()


def vec3_array(points: Tensor) -> wp.array:
    return wp.array(points.detach().cpu().reshape(-1, 3).numpy().astype(np.float32), dtype=wp.vec3)


class WaterViewer:
    """Draws a grid of seas, vessels and foil forces into a Newton viewer."""

    def __init__(
        self,
        viewer,
        num_envs: int,
        patch_size_m: float = 16.0,
        patch_resolution: int = 48,
        spacing_m: float | None = None,
        edge_fade: float = 0.0,
        sea_color: tuple[float, float, float] = SEA_COLOR,
        sea_opacity: float = 0.75,
        sea_roughness: float = 0.25,
    ) -> None:
        """`edge_fade` is the fraction of the patch width over which waves taper to flat at
        each border, so patches placed edge to edge (spacing = patch size) join without steps."""
        self.viewer = viewer
        self.num_envs = num_envs
        self.spacing_m = spacing_m or patch_size_m * 1.1
        columns = math.ceil(math.sqrt(num_envs))
        cells = torch.arange(num_envs)
        self.offsets = torch.stack(
            (
                (cells % columns) * self.spacing_m,
                -(cells // columns) * self.spacing_m,
                torch.zeros(num_envs),
            ),
            dim=-1,
        ).float()

        side = torch.linspace(-patch_size_m / 2, patch_size_m / 2, patch_resolution)
        grid_x, grid_y = torch.meshgrid(side, side, indexing="ij")
        self.patch = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1).reshape(-1, 3)
        self.sea_indices = self._tiled_triangles(patch_resolution, num_envs)
        self.registered: set[str] = set()
        self.sea_color = sea_color
        self.sea_opacity = sea_opacity
        self.sea_roughness = sea_roughness
        self.edge_weight = self._edge_weight(side, edge_fade)

    def look_at_grid(
        self, distance: float = 1.0, pitch_deg: float = -25.0, yaw_deg: float = 0.0
    ) -> None:
        """Aim the camera at the middle of the grid from `distance` grid-widths away;
        yaw 0 looks north (along +x), yaw 90 looks west from the east side."""
        center = self.offsets.mean(0)
        reach = (self.offsets[:, 0].max() - self.offsets[:, 0].min()).item() + self.spacing_m
        back = reach * distance
        yaw = math.radians(yaw_deg)
        position = wp.vec3(
            center[0].item() - back * math.cos(yaw),
            center[1].item() - back * math.sin(yaw),
            back * math.tan(math.radians(-pitch_deg)),
        )
        self.viewer.set_camera(position, pitch_deg, yaw_deg)

    def draw(
        self,
        t: Tensor,
        sea: Sea,
        vessel: Vessel | None = None,
        eta: Tensor | None = None,
        foil_loads: list[FoilLoads] | None = None,
        newton_per_m: float = 1500.0,
    ) -> None:
        followed = (
            torch.zeros(self.num_envs, 3)
            if eta is None
            else eta[:, :3].detach().cpu() * torch.tensor([1.0, 1.0, 0.0])
        )
        self.viewer.begin_frame(float(t.reshape(-1)[0]))
        self._draw_sea(t, sea, followed)
        if vessel is not None and eta is not None:
            self._draw_vessels(vessel, eta.detach().cpu(), followed)
        if foil_loads and eta is not None:
            self._draw_forces(foil_loads, eta.detach().cpu(), followed, newton_per_m)
        self.viewer.end_frame()

    def save_png(self, path: str) -> None:
        from PIL import Image

        Image.fromarray(self.viewer.get_frame().numpy()).save(path)

    def _draw_sea(self, t: Tensor, sea: Sea, followed: Tensor) -> None:
        device = sea.amplitude.device
        points = followed[:, None, :] + self.patch[None]
        surface = elevation(sea, points.to(device), t).cpu()
        local = self.patch[None].expand(self.num_envs, -1, -1).clone()
        local[..., 2] = -surface * self.edge_weight
        vertices = to_viewer_frame(local) + self.offsets[:, None]
        self.viewer.log_mesh(
            "sea",
            vec3_array(vertices),
            self.sea_indices,
            normals=vec3_array(self._grid_normals(vertices)),
            color=self.sea_color,
            roughness=self.sea_roughness,
            opacity=self.sea_opacity,
            dynamic=True,
            backface_culling=False,
        )

    def _grid_normals(self, vertices: Tensor) -> Tensor:
        n = round(math.sqrt(self.patch.shape[0]))
        grid = vertices.reshape(self.num_envs, n, n, 3)
        along_x, along_y = torch.gradient(grid, dim=(1, 2))
        return torch.nn.functional.normalize(torch.linalg.cross(along_x, along_y), dim=-1)

    def _draw_vessels(self, vessel: Vessel, eta: Tensor, followed: Tensor) -> None:
        if vessel.name not in self.registered:
            mesh = vessel.mesh
            indices = wp.array(mesh.triangles.reshape(-1).numpy().astype(np.int32), dtype=wp.int32)
            palette, uvs = palette_texture(mesh.colors)
            self.viewer.log_mesh(
                vessel.name,
                vec3_array(to_viewer_frame(mesh.vertices)),
                indices,
                uvs=wp.array(uvs.numpy(), dtype=wp.vec2),
                texture=palette,
                hidden=True,
            )
            self.registered.add(vessel.name)
        pose = Pose.from_eta(eta)
        flip = torch.diag(FLIP)
        rotation = flip @ pose.rotation @ flip
        position = to_viewer_frame(eta[:, :3] - followed) + self.offsets
        transforms = torch.cat((position, quaternion_xyzw(rotation)), dim=-1)
        colors = torch.tensor(HULL_COLOR).expand(self.num_envs, 3)
        textured = torch.tensor(VESSEL_MATERIAL).expand(self.num_envs, 4)
        self.viewer.log_instances(
            f"{vessel.name}_fleet",
            vessel.name,
            wp.array(transforms.numpy().astype(np.float32), dtype=wp.transform),
            vec3_array(torch.ones(self.num_envs, 3)),
            vec3_array(colors),
            wp.array(textured.numpy().astype(np.float32), dtype=wp.vec4),
        )

    def _draw_forces(
        self, foil_loads: list[FoilLoads], eta: Tensor, followed: Tensor, newton_per_m: float
    ) -> None:
        rotation = Pose.from_eta(eta).rotation
        starts, ends = [], []
        for loads in foil_loads:
            points = loads.points.detach().cpu() - followed[:, None]
            force_world = loads.force.detach().cpu() @ rotation.transpose(1, 2)
            starts.append(to_viewer_frame(points) + self.offsets[:, None])
            ends.append(
                to_viewer_frame(points + force_world / newton_per_m) + self.offsets[:, None]
            )
        self.viewer.log_arrows(
            "foil_forces",
            vec3_array(torch.cat(starts, 1)),
            vec3_array(torch.cat(ends, 1)),
            FORCE_COLOR,
        )

    @staticmethod
    def _edge_weight(side: Tensor, edge_fade: float) -> Tensor:
        """Smoothstep from 0 at the patch border to 1 `edge_fade` of the width inside, [P]."""
        if edge_fade <= 0:
            return torch.ones(side.numel() ** 2)
        width = (side[-1] - side[0]).item()
        inside = ((side - side[0]).minimum(side[-1] - side) / (edge_fade * width)).clamp(0, 1)
        ramp = inside * inside * (3 - 2 * inside)
        return (ramp[:, None] * ramp[None, :]).reshape(-1)

    @staticmethod
    def _tiled_triangles(resolution: int, copies: int) -> wp.array:
        i, j = torch.meshgrid(
            torch.arange(resolution - 1), torch.arange(resolution - 1), indexing="ij"
        )
        corner = (i * resolution + j).reshape(-1)
        quads = torch.stack(
            (corner, corner + resolution, corner + resolution + 1, corner + 1), dim=-1
        )
        triangles = torch.cat((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
        tiled = triangles[None] + torch.arange(copies)[:, None, None] * resolution**2
        return wp.array(tiled.reshape(-1).numpy().astype(np.int32), dtype=wp.int32)
