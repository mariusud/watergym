"""Strip-theory lift and drag on foils, struts and rudders.

Each foil is cut into spanwise strips. A strip sees the water velocity relative to itself,
body motion and wave orbital velocity included, and produces

    L = 1/2 rho |U|^2 A C_L(alpha),   D = 1/2 rho |U|^2 A C_D(alpha)

with C_L reduced near the free surface and collapsed when the foil is ventilated.

The ventilation thresholds below are placeholders. Harwood et al. map the regimes but give
no numbers, so every threshold is a Foil field meant to be
domain-randomized.
"""

import math
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from watergym.geometry import Vec3
from watergym.hydrostatics import WATER_DENSITY, submerged_fraction
from watergym.rigid_body import Pose
from watergym.waves import GRAVITY, Sea, Water, water_at

# A horizontal strip has no height, so without this ramp its lift would switch on in one
# step as it crosses the surface. Lift fades in over this many chords of depth instead.
WETTING_RAMP_CHORDS = 0.1


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
    # None: Helmbold's slope for the geometric aspect ratio span / chord.
    lift_slope: float | None = None
    stall_angle: float = math.radians(12)
    zero_lift_drag: float = 0.008
    oswald_efficiency: float = 0.9
    flap_effectiveness: float = 0.0
    max_flap: float = 0.0
    # Ventilation, all placeholders to randomize: lift and profile drag multipliers while
    # ventilated, then the onset and washout thresholds of update_ventilation.
    ventilated_lift_ratio: float = 0.25
    ventilated_drag_ratio: float = 1.5
    onset_froude_scale: float = 1.0
    onset_depth_chords: float = 1.0
    ventilation_onset_angle: float = math.radians(10)
    washout_depth_chords: float = 1.0
    ventilation_washout_angle: float = math.radians(4)
    washout_time_s: float = 0.5
    num_strips: int = 8
    strip_centers: Tensor = field(init=False)
    span_vector: Tensor = field(init=False)
    chord_vector: Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.lift_slope is None:
            self.lift_slope = helmbold_lift_slope(self.aspect_ratio)
        offsets = (torch.arange(self.num_strips) + 0.5) / self.num_strips - 0.5
        self.span_vector = torch.tensor(self.span_axis)
        self.chord_vector = torch.tensor(self.chord_axis)
        self.strip_centers = (
            torch.tensor(self.position) + offsets[:, None] * self.span_m * self.span_vector
        )

    def to(self, device: str | torch.device) -> "Foil":
        moved = replace(self)
        moved.strip_centers = self.strip_centers.to(device)
        moved.span_vector = self.span_vector.to(device)
        moved.chord_vector = self.chord_vector.to(device)
        return moved

    @property
    def aspect_ratio(self) -> float:
        return self.span_m / self.chord_m

    @property
    def is_horizontal(self) -> bool:
        """Span within 45 degrees of horizontal: a foil, not a strut."""
        return abs(self.span_axis[2]) < math.sqrt(0.5)

    @property
    def onset_depth_froude(self) -> float:
        """Fr_h above which a near-surface strip can ventilate: AR^-1/2 (plan section 3.5,
        from arXiv:2503.18015), times a scale to randomize."""
        return self.onset_froude_scale * self.aspect_ratio**-0.5

    @property
    def has_flap(self) -> bool:
        return self.max_flap > 0

    @property
    def strip_area(self) -> float:
        return self.span_m * self.chord_m / self.num_strips

    @property
    def immersion_band(self) -> float:
        """Vertical distance over which a strip goes from dry to fully wet."""
        return (
            abs(self.span_axis[2]) * self.span_m / self.num_strips
            + WETTING_RAMP_CHORDS * self.chord_m
        )


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
    water: Water | None = None,
) -> FoilLoads:
    """Strip forces for a batch of vessels. `flap` [envs] in rad, `ventilated` [envs] bool."""
    r, span, chord = foil.strip_centers, foil.span_vector, foil.chord_vector
    normal = torch.linalg.cross(span, chord)

    points = pose.to_world(r)
    water = water or water_at(sea, points, t)
    strip_velocity = nu[:, None, :3] + torch.linalg.cross(nu[:, None, 3:], r[None])
    flow = pose.to_body(water.velocity) - strip_velocity
    flow = flow - (flow @ span)[..., None] * span
    speed = flow.norm(dim=-1).clamp(min=1e-6)

    alpha = torch.atan2(flow @ normal, -(flow @ chord))
    alpha = alpha + foil.incidence + foil.flap_effectiveness * flap[:, None]
    depth = points[..., 2] + water.elevation

    lift_coefficient = foil.lift_slope * alpha.clamp(-foil.stall_angle, foil.stall_angle)
    # Image-vortex loss for horizontal foils only. A vertical strut loses lift near the
    # surface only through its dry strips, which carry nothing (wet_area below).
    if foil.is_horizontal:
        lift_coefficient = lift_coefficient * free_surface_factor(depth / foil.chord_m)
    vented = ventilated[:, None]
    lift_coefficient = lift_coefficient * torch.where(vented, foil.ventilated_lift_ratio, 1.0)
    profile_drag = foil.zero_lift_drag * torch.where(vented, foil.ventilated_drag_ratio, 1.0)
    drag_coefficient = profile_drag + lift_coefficient**2 / (
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
    r = foil.strip_centers.expand_as(loads.force)
    moment = torch.linalg.cross(r, loads.force)
    return torch.cat((loads.force.sum(1), moment.sum(1)), dim=-1)


def update_ventilation(
    foil: Foil,
    ventilated: Tensor,
    loads: FoilLoads,
    wetting_time: Tensor | None = None,
    dt: float = 0.0,
) -> Tensor:
    """Two-state ventilation automaton (plan section 3.5), decided strip by strip.

    Onset: any wet strip within `onset_depth_chords` of the surface runs at a depth Froude
    number U / sqrt(g h) above `onset_depth_froude` and an angle above the onset angle. Air
    is drawn down from the surface, so the shallowest strips decide, not the foil's mean.

    Washout: every wet strip is below the washout angle and, for a horizontal foil, the
    shallowest strip is deeper than `washout_depth_chords`, which cuts off the air. A
    surface-piercing strut always has an air path, so only its angle can wash it out. These
    conditions must hold for `washout_time_s`, counted in `wetting_time` [envs], which is
    advanced in place. Without a `wetting_time` the foil re-wets as soon as they hold.
    """
    wet = loads.depth > -foil.immersion_band / 2
    alpha = loads.alpha.abs()
    depth_froude = loads.speed / torch.sqrt(GRAVITY * loads.depth.clamp(min=1e-3))
    near_surface = loads.depth < foil.onset_depth_chords * foil.chord_m
    onset = (
        wet
        & near_surface
        & (depth_froude > foil.onset_depth_froude)
        & (alpha > foil.ventilation_onset_angle)
    ).any(-1)

    calm = ((alpha < foil.ventilation_washout_angle) | ~wet).all(-1)
    if foil.is_horizontal:
        shallowest = loads.depth.amin(-1)
        calm = calm & (shallowest > foil.washout_depth_chords * foil.chord_m)

    if wetting_time is None:
        rewetted = calm
    else:
        wetting_time.copy_(torch.where(ventilated & calm, wetting_time + dt, 0.0))
        rewetted = wetting_time >= foil.washout_time_s
    return torch.where(ventilated, ~rewetted, onset)
