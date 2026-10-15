"""Wind: a mean speed and direction per environment plus gusts, and the load it puts on the
part of the vessel above the water.

The wind blows toward `direction` (rad from north toward east), the same convention as the
wave heading. Its speed at 10 m is the mean U10 plus a sum of sinusoids drawn from the
Frøya spectrum, then scaled to the vessel's height with a power law:

    U(z, t) = (U10 + sum_i a_i cos(omega_i t + phi_i)) (z / 10)^alpha

The load follows Fossen's wind model with simple cosine and sine coefficients (Handbook of
Marine Craft Hydrodynamics and Motion Control, 2nd ed., 2021, sec. 10.1). The coefficients
are an approximation for a generic hull, not a measured set.
"""

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from watergym.geometry import Mesh
from watergym.rigid_body import Pose, rotation_matrix

AIR_DENSITY = 1.225
REFERENCE_HEIGHT_M = 10.0
# Power-law exponent of the mean wind profile over open sea, DNV-RP-C205 sec. 2.3.2.
PROFILE_EXPONENT = 0.12
# Gust band: from one hour, the averaging time of U10 in the Frøya spectrum, to 0.2 s.
GUST_BAND_HZ = (1 / 3600, 5.0)


@dataclass
class WindState:
    """Mean wind at 10 m: `speed` in m/s toward `direction_rad`. `gusts=False` holds it
    constant. Zero speed means no wind at all."""

    speed: float = 0.0
    direction_rad: float = 0.0
    gusts: bool = True
    num_components: int = 64


@dataclass
class Wind:
    """Per environment: mean speed and direction at 10 m [num_envs], gust components
    [num_envs, num_components] (none for a constant wind)."""

    speed: Tensor
    direction: Tensor
    amplitude: Tensor
    omega: Tensor
    phase: Tensor

    @property
    def num_envs(self) -> int:
        return self.speed.shape[0]

    def subset(self, index: slice | list[int] | Tensor) -> "Wind":
        return Wind(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def replace(self, env_ids: Tensor, other: "Wind") -> None:
        for f in fields(self):
            getattr(self, f.name)[env_ids] = getattr(other, f.name)


@dataclass
class Windage:
    """The vessel above the water, as Fossen's wind model needs it. `frontal_z_m` and
    `lateral_z_m` are the body-frame heights (z down) where the X and Y forces act;
    `height_m` is where the wind speed is read, above the mean water. c_x, c_y and c_n are
    Fossen's coefficients, inside his ranges 0.50-0.90, 0.70-0.95 and 0.05-0.20."""

    frontal_area_m2: float
    lateral_area_m2: float
    length_m: float
    frontal_z_m: float
    lateral_z_m: float
    height_m: float
    c_x: float = 0.7
    c_y: float = 0.85
    c_n: float = 0.1


def froya_spectrum(frequency_hz: Tensor, speed: Tensor, height_m: float = 10.0) -> Tensor:
    """Frøya gust spectrum S(f) [m^2/s^2 per Hz] for a one-hour mean speed U10 at 10 m,
    DNV-RP-C205 sec. 2.3.5 (also NORSOK N-003):

        S(f) = 320 (U10/10)^2 (z/10)^0.45 / (1 + f~^n)^(5/(3n)),  n = 0.468
        f~   = 172 f (z/10)^(2/3) (U10/10)^(-3/4)
    """
    n = 0.468
    scaled_f = 172 * frequency_hz * (height_m / 10) ** (2 / 3) / (speed / 10) ** 0.75
    return 320 * (speed / 10) ** 2 * (height_m / 10) ** 0.45 / (1 + scaled_f**n) ** (5 / (3 * n))


def make_wind(
    num_envs: int,
    speed: float | Tensor,
    direction_rad: float | Tensor = 0.0,
    gusts: bool = True,
    num_components: int = 64,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> Wind:
    """Draw a wind per environment with independent random gust phases.

    The gust band GUST_BAND_HZ is cut into log-spaced bins. Each amplitude follows
    a_i = sqrt(2 S(f_i) df_i) at the bin's geometric centre, and each component oscillates at
    a random frequency inside its bin, as the waves do.
    """
    generator = generator or torch.Generator().manual_seed(0)

    def per_env(value: float | Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device).expand(num_envs)

    speed, direction = per_env(speed).clone(), per_env(direction_rad).clone()
    count = num_components if gusts else 0
    low, high = GUST_BAND_HZ
    edges = torch.logspace(math.log10(low), math.log10(high), count + 1, device=device)
    centers = (edges[:-1] * edges[1:]).sqrt()
    widths = edges[1:] - edges[:-1]
    amplitude = torch.sqrt(2 * froya_spectrum(centers, speed[:, None].clamp(min=1e-6)) * widths)
    amplitude = torch.where(speed[:, None] > 0, amplitude, 0.0)
    within = torch.rand(num_envs, count, generator=generator).to(device)
    phase = 2 * math.pi * torch.rand(num_envs, count, generator=generator).to(device)
    return Wind(
        speed=speed,
        direction=direction,
        amplitude=amplitude,
        omega=2 * math.pi * (edges[:-1] + within * widths),
        phase=phase,
    )


def wind_speed(wind: Wind, t: Tensor, height_m: float) -> Tensor:
    """U(z, t) = (U10 + sum_i a_i cos(omega_i t + phi_i)) (z / 10)^alpha [num_envs]."""
    t = torch.as_tensor(t, dtype=wind.speed.dtype, device=wind.speed.device).reshape(-1, 1)
    gust = (wind.amplitude * torch.cos(wind.omega * t + wind.phase)).sum(-1)
    profile = (max(height_m, 0.0) / REFERENCE_HEIGHT_M) ** PROFILE_EXPONENT
    return (wind.speed + gust) * profile


def wind_force(wind: Wind, windage: Windage, t: Tensor, pose: Pose, nu: Tensor) -> Tensor:
    """Wind load tau [envs, 6] in the body frame, Fossen (2021) sec. 10.1.

    The relative velocity is u_r = u - u_w, v_r = v - v_w, with (u_w, v_w) the wind velocity
    in the body frame, V_r^2 = u_r^2 + v_r^2 and angle of attack gamma = -atan2(v_r, u_r):

        X = 1/2 rho_a V_r^2 C_X A_F,       C_X = -c_x cos(gamma)
        Y = 1/2 rho_a V_r^2 C_Y A_L,       C_Y =  c_y sin(gamma)
        N = 1/2 rho_a V_r^2 C_N A_L L,     C_N =  c_n sin(2 gamma)

    written without the angle as X = -1/2 rho_a c_x A_F V_r u_r, Y = -1/2 rho_a c_y A_L V_r v_r
    and N = -rho_a c_n A_L L u_r v_r. X and Y act at the area centroids, which gives the
    roll moment K = -z_L Y and pitch moment M = z_F X.
    """
    speed = wind_speed(wind, t, windage.height_m)
    north, east = speed * torch.cos(wind.direction), speed * torch.sin(wind.direction)
    wind_body = pose.rotation[:, 0, :] * north[:, None] + pose.rotation[:, 1, :] * east[:, None]
    u_r, v_r = nu[:, 0] - wind_body[:, 0], nu[:, 1] - wind_body[:, 1]
    relative_speed = torch.sqrt(u_r**2 + v_r**2)
    half_rho = 0.5 * AIR_DENSITY
    x = -half_rho * windage.c_x * windage.frontal_area_m2 * relative_speed * u_r
    y = -half_rho * windage.c_y * windage.lateral_area_m2 * relative_speed * v_r
    n = -2 * half_rho * windage.c_n * windage.lateral_area_m2 * windage.length_m * u_r * v_r
    zero = torch.zeros_like(x)
    return torch.stack((x, y, zero, -windage.lateral_z_m * y, windage.frontal_z_m * x, n), dim=-1)


def covered_cells(corners: Tensor, cell_m: float, top: float) -> Tensor:
    """Centres [C, 2] of the grid cells of size cell_m whose centre lies inside one of the
    triangles corners [T, 3, 2], keeping only cells with second coordinate below `top`."""
    low, high = corners.reshape(-1, 2).amin(0), corners.reshape(-1, 2).amax(0)
    if high[1] > top:
        high = torch.stack((high[0], torch.tensor(top)))
    if (high <= low).any():
        return torch.zeros(0, 2)
    a = torch.arange(low[0] + cell_m / 2, high[0], cell_m)
    b = torch.arange(low[1] + cell_m / 2, high[1], cell_m)
    grid = torch.stack(torch.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
    covered = torch.zeros(len(grid), dtype=torch.bool)

    def side(p: Tensor, q: Tensor) -> Tensor:
        """(q - p) x (grid - p) for every triangle edge p -> q and grid point [t, G]."""
        edge, offset = q - p, grid[None] - p[:, None]
        return edge[:, None, 0] * offset[..., 1] - edge[:, None, 1] * offset[..., 0]

    p0, p1, p2 = corners.unbind(1)
    doubled_area = (p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0]
    keep = doubled_area.abs() > 1e-12
    for chunk in torch.cat((p0, p1, p2), -1)[keep].split(64):
        q0, q1, q2 = chunk[:, :2], chunk[:, 2:4], chunk[:, 4:]
        e0, e1, e2 = side(q0, q1), side(q1, q2), side(q2, q0)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        covered |= inside.any(0)
    return grid[covered]


def windage_from_mesh(
    mesh: Mesh,
    initial_eta: tuple[float, ...],
    resolution: int = 200,
) -> Windage:
    """Frontal and lateral projected areas of the mesh above the mean water at the initial
    pose, by rasterizing its silhouette on a grid of `resolution` cells along the longest
    side. L is the length of the lateral silhouette."""
    phi, theta = torch.tensor(initial_eta[3]), torch.tensor(initial_eta[4])
    rotation = rotation_matrix(phi, theta, torch.tensor(0.0))
    points = mesh.vertices @ rotation.T
    waterline_z = -initial_eta[2]
    cell_m = (points.amax(0) - points.amin(0)).max().item() / resolution
    corners = points[mesh.triangles]
    frontal = covered_cells(corners[..., [1, 2]], cell_m, waterline_z)
    lateral = covered_cells(corners[..., [0, 2]], cell_m, waterline_z)
    if len(lateral) == 0:
        return Windage(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lateral_z = lateral[:, 1].mean().item()
    return Windage(
        frontal_area_m2=len(frontal) * cell_m**2,
        lateral_area_m2=len(lateral) * cell_m**2,
        length_m=(lateral[:, 0].amax() - lateral[:, 0].amin()).item() + cell_m,
        frontal_z_m=frontal[:, 1].mean().item() if len(frontal) else lateral_z,
        lateral_z_m=lateral_z,
        height_m=waterline_z - lateral_z,
    )
