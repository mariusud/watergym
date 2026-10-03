"""Strip-theory lift and drag on foils, struts and rudders.

Each foil is cut into spanwise strips. A strip sees the water velocity relative to itself,
body motion and wave orbital velocity included, and produces

    L = 1/2 rho |U|^2 A C_L(alpha),   D = 1/2 rho |U|^2 A C_D(alpha)

with C_L reduced near the free surface and collapsed when the foil is ventilated.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from watergym.geometry import Vec3
from watergym.hydrostatics import WATER_DENSITY, submerged_fraction
from watergym.rigid_body import Pose
from watergym.waves import GRAVITY, Sea, elevation, orbital_velocity


def helmbold_lift_slope(aspect_ratio: float) -> float:
    """Finite-span lift-curve slope per radian, Helmbold's low-aspect-ratio form of lifting
    line. Prandtl's 2 pi AR / (AR + 2) overpredicts by 20 % at AR = 2."""
    return math.pi * aspect_ratio / (1 + math.sqrt(1 + (aspect_ratio / 2) ** 2))


def free_surface_factor(depth_over_chord: Tensor) -> Tensor:
    """C_L / C_L,deep at high Froude number: a vortex and its negative image at 2h
    (Faltinsen, Hydrodynamics of High-Speed Marine Vehicles, 2005). 0.5 at the surface."""
    x2 = 16 * depth_over_chord.clamp(min=0) ** 2
    return (1 + x2) / (2 + x2)


@dataclass
class Foil:
    """A straight lifting surface. Axes are body-frame unit vectors; `position` is the
    centre of the span. A horizontal foil lifts up (-z); a vertical strut lifts to starboard."""

    name: str
    position: Vec3
    span_m: float
    chord_m: float
    span_axis: Vec3 = (0.0, 1.0, 0.0)
    chord_axis: Vec3 = (1.0, 0.0, 0.0)
    incidence: float = 0.0
    lift_slope: float = 2 * math.pi
    stall_angle: float = math.radians(12)
    zero_lift_drag: float = 0.008
    oswald_efficiency: float = 0.9
    flap_effectiveness: float = 0.0
    max_flap: float = 0.0
    ventilated_lift_ratio: float = 0.25
    ventilation_onset_angle: float = math.radians(10)
    ventilation_washout_angle: float = math.radians(4)
    num_strips: int = 8
    strip_centers: Tensor = field(init=False)

    def __post_init__(self) -> None:
        offsets = (torch.arange(self.num_strips) + 0.5) / self.num_strips - 0.5
        span_axis = torch.tensor(self.span_axis)
        self.strip_centers = (
            torch.tensor(self.position) + offsets[:, None] * self.span_m * span_axis
        )

    @property
    def aspect_ratio(self) -> float:
        return self.span_m / self.chord_m

    @property
    def has_flap(self) -> bool:
        return self.max_flap > 0

    @property
    def strip_area(self) -> float:
        return self.span_m * self.chord_m / self.num_strips

    @property
    def immersion_band(self) -> float:
        """Vertical distance over which a strip goes from dry to fully wet."""
        return abs(self.span_axis[2]) * self.span_m / self.num_strips + 0.1 * self.chord_m


@dataclass
class FoilLoads:
    """Per-strip state of one foil: world positions [envs, S, 3], body-frame force
    [envs, S, 3], angle of attack, depth below the local surface and flow speed [envs, S]."""

    points: Tensor
    force: Tensor
    alpha: Tensor
    depth: Tensor
    speed: Tensor


def foil_loads(
    foil: Foil,
    sea: Sea,
    t: Tensor,
    pose: Pose,
    nu: Tensor,
    flap: Tensor,
    ventilated: Tensor,
    density: float = WATER_DENSITY,
) -> FoilLoads:
    """Strip forces for a batch of vessels. `flap` [envs] in rad, `ventilated` [envs] bool."""
    device = nu.device
    r = foil.strip_centers.to(device)
    span = torch.tensor(foil.span_axis, device=device)
    chord = torch.tensor(foil.chord_axis, device=device)
    normal = torch.linalg.cross(span, chord)

    points = pose.to_world(r)
    strip_velocity = nu[:, None, :3] + torch.linalg.cross(nu[:, None, 3:], r[None])
    flow = pose.to_body(orbital_velocity(sea, points, t)) - strip_velocity
    flow = flow - (flow @ span)[..., None] * span
    speed = flow.norm(dim=-1).clamp(min=1e-6)

    alpha = torch.atan2(flow @ normal, -(flow @ chord))
    alpha = alpha + foil.incidence + foil.flap_effectiveness * flap[:, None]
    depth = points[..., 2] + elevation(sea, points, t)

    lift_coefficient = foil.lift_slope * alpha.clamp(-foil.stall_angle, foil.stall_angle)
    lift_coefficient = lift_coefficient * free_surface_factor(depth / foil.chord_m)
    lift_coefficient = torch.where(
        ventilated[:, None], foil.ventilated_lift_ratio * lift_coefficient, lift_coefficient
    )
    drag_coefficient = foil.zero_lift_drag + lift_coefficient**2 / (
        math.pi * foil.oswald_efficiency * foil.aspect_ratio
    )

    dynamic_pressure = 0.5 * density * speed**2
    wet_area = foil.strip_area * submerged_fraction(depth, foil.immersion_band)
    lift_direction = torch.linalg.cross(flow, span.expand_as(flow)) / speed[..., None]
    drag_direction = flow / speed[..., None]
    force = (dynamic_pressure * wet_area)[..., None] * (
        lift_coefficient[..., None] * lift_direction + drag_coefficient[..., None] * drag_direction
    )
    return FoilLoads(points, force, alpha, depth, speed)


def foil_wrench(foil: Foil, loads: FoilLoads) -> Tensor:
    """Total force and moment about the CG in the body frame [envs, 6]."""
    r = foil.strip_centers.to(loads.force.device).expand_as(loads.force)
    moment = torch.linalg.cross(r, loads.force)
    return torch.cat((loads.force.sum(1), moment.sum(1)), dim=-1)


def update_ventilation(foil: Foil, ventilated: Tensor, loads: FoilLoads) -> Tensor:
    """Two-state ventilation with hysteresis in angle of attack.

    Air reaches the suction side when the foil is near the surface, fast (depth Froude
    number U / sqrt(g h) above AR^-1/2, plan section 3.5) and loaded past the onset angle.
    The flow re-attaches only once the angle drops below the lower washout angle.
    """
    alpha = loads.alpha.abs().mean(-1)
    depth = loads.depth.mean(-1).clamp(min=1e-3)
    depth_froude = loads.speed.mean(-1) / torch.sqrt(GRAVITY * depth)
    near_surface = depth < foil.chord_m
    onset = (
        near_surface
        & (depth_froude > foil.aspect_ratio**-0.5)
        & (alpha > foil.ventilation_onset_angle)
    )
    washout = alpha < foil.ventilation_washout_angle
    return (ventilated | onset) & ~washout
