"""Irregular seas as a sum of linear (Airy) wave components, one sea per environment.

Frames follow Fossen: NED world, z points down. The wave elevation zeta is positive up,
so the instantaneous surface sits at z = -zeta and a point is wet when z + zeta > 0.
"""

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
from torch import Tensor

GRAVITY = 9.81
# Largest [envs, points, components] tensor water_at builds: 64 MB in float32. Splitting
# further costs speed on CPU and MPS; above it the peak memory grows with no speed gain.
MAX_CHUNK_ELEMENTS = 2**24


@dataclass
class Sea:
    """Wave components per environment, each tensor shaped [num_envs, num_components]."""

    amplitude: Tensor
    omega: Tensor
    wavenumber: Tensor
    direction: Tensor
    phase: Tensor

    @property
    def num_envs(self) -> int:
        return self.amplitude.shape[0]

    def subset(self, index: slice | list[int] | Tensor) -> "Sea":
        """The seas of the selected envs only: a slice, a list of env ids or a tensor of ids."""
        return Sea(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def replace(self, env_ids: Tensor, other: "Sea") -> None:
        for name in ("amplitude", "omega", "wavenumber", "direction", "phase"):
            getattr(self, name)[env_ids] = getattr(other, name)


def jonswap(omega: Tensor, hs: Tensor, tp: Tensor, gamma: float = 3.3) -> Tensor:
    """JONSWAP spectral density S(omega) [m^2 s/rad], DNV-RP-C205 eq. 3.5.5.1."""
    omega_p = 2 * math.pi / tp
    pierson_moskowitz = (
        5 / 16 * hs**2 * omega_p**4 * omega**-5 * torch.exp(-5 / 4 * (omega / omega_p) ** -4)
    )
    sigma = torch.where(omega <= omega_p, 0.07, 0.09)
    peak_shape = gamma ** torch.exp(-((omega - omega_p) ** 2) / (2 * sigma**2 * omega_p**2))
    return (1 - 0.287 * math.log(gamma)) * pierson_moskowitz * peak_shape


def sample_directions(
    num_envs: int,
    num_components: int,
    heading_rad: Tensor,
    spreading: float | None,
    generator: torch.Generator,
) -> Tensor:
    """Directions waves travel toward, measured from north (x) toward east (y).

    `spreading` is the s of the cos-2s distribution D(theta) ~ cos^(2s)((theta - heading_rad) / 2);
    None gives long-crested waves.
    """
    heading_rad = heading_rad[:, None].expand(num_envs, num_components)
    if spreading is None:
        return heading_rad.clone()
    offsets = torch.linspace(-math.pi, math.pi, 361, device=heading_rad.device)
    weights = torch.cos(offsets / 2).abs() ** (2 * spreading)
    picks = torch.multinomial(
        weights.cpu(), num_envs * num_components, replacement=True, generator=generator
    )
    return heading_rad + offsets[picks.to(heading_rad.device)].view(num_envs, num_components)


def make_sea(
    num_envs: int,
    hs: float | Tensor,
    tp: float | Tensor,
    heading_rad: float | Tensor = 0.0,
    spreading: float | None = None,
    gamma: float = 3.3,
    num_components: int = 64,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> Sea:
    """Draw a JONSWAP sea per environment with independent random phases.

    The band [0.5, 4] omega_p is cut into equal bins. Each amplitude follows
    a_i = sqrt(2 S(omega_i) d_omega) at the bin centre, and each component oscillates at a
    random frequency inside its bin so the record does not repeat.
    """
    generator = generator or torch.Generator().manual_seed(0)

    def per_env(value: float | Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device).expand(num_envs)

    hs, tp, heading_rad = per_env(hs), per_env(tp), per_env(heading_rad)

    def uniform(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator).to(device)

    omega_p = 2 * math.pi / tp[:, None]
    bin_edges = torch.linspace(0.5, 4.0, num_components + 1, device=device)
    bin_width = (bin_edges[1] - bin_edges[0]) * omega_p
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2 * omega_p
    amplitude = torch.sqrt(2 * jonswap(bin_center, hs[:, None], tp[:, None], gamma) * bin_width)
    omega = bin_center + (uniform(num_envs, num_components) - 0.5) * bin_width
    return Sea(
        amplitude=amplitude,
        omega=omega,
        wavenumber=omega**2 / GRAVITY,
        direction=sample_directions(num_envs, num_components, heading_rad, spreading, generator),
        phase=2 * math.pi * uniform(num_envs, num_components),
    )


def regular_wave(
    num_envs: int, amplitude: float, period: float, heading_rad: float = 0.0, device: str = "cpu"
) -> Sea:
    """A single sinusoidal wave train, the textbook test case."""

    def full(value: float) -> Tensor:
        return torch.full((num_envs, 1), value, dtype=torch.float32, device=device)

    omega = 2 * math.pi / period
    return Sea(
        amplitude=full(amplitude),
        omega=full(omega),
        wavenumber=full(omega**2 / GRAVITY),
        direction=full(heading_rad),
        phase=full(0.0),
    )


class Water(NamedTuple):
    """The sea at points: elevation [envs, points], velocity and acceleration [envs, points, 3]."""

    elevation: Tensor
    velocity: Tensor
    acceleration: Tensor


def water_at(sea: Sea, points: Tensor, t: Tensor | float) -> Water:
    """Elevation (positive up), deep-water Airy velocity and local acceleration d(velocity)/dt.

    Per component, with phase theta = k (x cos(beta) + y sin(beta)) - omega t + phi and
    decay d = e^(-k z):

        elevation        a cos(theta)
        velocity         a omega d   (cos(theta) cos(beta), cos(theta) sin(beta), -sin(theta))
        acceleration     a omega^2 d (sin(theta) cos(beta), sin(theta) sin(beta),  cos(theta))

    The [envs, points, components] work is the phase, its cos and sin, the decay, and one
    multiply-and-sum per output. Those tensors set the memory use, so points are taken in
    chunks of at most MAX_CHUNK_ELEMENTS per tensor. Points above the mean level use the
    mean-level velocity and acceleration (constant extrapolation).
    """
    # Under torch.compile the fused kernels never hold these tensors, and chunks would push
    # a Metal kernel past its 31 buffer arguments.
    points_per_chunk = max(1, MAX_CHUNK_ELEMENTS // sea.amplitude.numel())
    if points.shape[-2] > points_per_chunk and not torch.compiler.is_compiling():
        chunks = [water_at(sea, part, t) for part in points.split(points_per_chunk, dim=-2)]
        return Water(*(torch.cat(parts, dim=1) for parts in zip(*chunks, strict=True)))

    t = torch.as_tensor(t, dtype=points.dtype, device=points.device).reshape(-1, 1)
    cos_beta, sin_beta = torch.cos(sea.direction), torch.sin(sea.direction)
    k = sea.wavenumber[:, None]
    x, y, z = points[..., 0, None], points[..., 1, None], points[..., 2, None]
    time_phase = (sea.phase - sea.omega * t)[:, None]
    theta = time_phase.addcmul(x, k * cos_beta[:, None]).addcmul_(y, k * sin_beta[:, None])
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    del theta  # the [envs, points, components] tensors dominate memory: free them early
    decay = (-k * z.clamp(min=0)).exp_()
    wet_sin = sin_theta.mul_(decay)
    wet_cos = decay.mul_(cos_theta)

    speed = sea.amplitude * sea.omega
    accel = speed * sea.omega

    def total(per_component: Tensor, weight: Tensor) -> Tensor:
        return (per_component * weight[:, None]).sum(-1)

    return Water(
        elevation=total(cos_theta, sea.amplitude),
        velocity=torch.stack(
            (
                total(wet_cos, speed * cos_beta),
                total(wet_cos, speed * sin_beta),
                -total(wet_sin, speed),
            ),
            dim=-1,
        ),
        acceleration=torch.stack(
            (
                total(wet_sin, accel * cos_beta),
                total(wet_sin, accel * sin_beta),
                total(wet_cos, accel),
            ),
            dim=-1,
        ),
    )


def elevation(sea: Sea, points: Tensor, t: Tensor | float) -> Tensor:
    """Surface elevation eta (positive up) at the points' (x, y), shaped [envs, points]."""
    return water_at(sea, points, t).elevation


def orbital_velocity(sea: Sea, points: Tensor, t: Tensor | float) -> Tensor:
    """Water velocity in NED at points [envs, points, 3], deep-water Airy theory."""
    return water_at(sea, points, t).velocity


def orbital_acceleration(sea: Sea, points: Tensor, t: Tensor | float) -> Tensor:
    """Local water acceleration d(velocity)/dt in NED at points [envs, points, 3]."""
    return water_at(sea, points, t).acceleration


def significant_wave_height(sea: Sea) -> Tensor:
    """Hs = 4 sqrt(m0), with m0 the variance of the realized components."""
    return 4 * torch.sqrt((sea.amplitude**2 / 2).sum(-1))
