"""A vessel is a rigid body plus the things that push on it: hull volume, foils, thrusters.

    tau = gravity + hydrostatics + foils + thrusters
    nu_dot = (M_RB + M_A)^-1 (tau - C(nu) nu - D(nu) nu)

The action vector is [thruster commands..., flap commands...], each in [-1, 1].
"""

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from watergym.foils import Foil, FoilLoads, foil_loads, foil_wrench
from watergym.geometry import Mesh, Vec3, VolumeSamples
from watergym.hydrostatics import WATER_DENSITY, hydrostatic_force
from watergym.rigid_body import Derivative, Pose, RigidBody, acceleration, kinematics
from watergym.waves import GRAVITY, Sea, Water, water_at

Vec6 = tuple[float, float, float, float, float, float]


@dataclass
class Thruster:
    position: Vec3
    direction: Vec3
    max_force_n: float
    force_vector: Tensor = field(init=False)
    position_vector: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.force_vector = self.max_force_n * torch.tensor(self.direction)
        self.position_vector = torch.tensor(self.position)

    def to(self, device: str | torch.device) -> "Thruster":
        moved = replace(self)
        moved.force_vector = self.force_vector.to(device)
        moved.position_vector = self.position_vector.to(device)
        return moved


@dataclass
class Vessel:
    """`hull` must put the centre of buoyancy above the centre of gravity (body origin), or an
    un-actuated vessel tumbles: unequal added masses give a Munk moment that tips it over."""

    name: str
    body: RigidBody
    hull: VolumeSamples
    mesh: Mesh
    foils: list[Foil] = field(default_factory=list)
    thrusters: list[Thruster] = field(default_factory=list)
    free_dofs: tuple[bool, ...] = (True,) * 6
    initial_eta: Vec6 = (0.0,) * 6
    initial_nu: Vec6 = (0.0,) * 6
    density: float = WATER_DENSITY
    free_dof_mask: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.free_dof_mask = torch.tensor(self.free_dofs, dtype=torch.float32)

    @property
    def flapped_foils(self) -> list[Foil]:
        return [foil for foil in self.foils if foil.has_flap]

    @property
    def num_actions(self) -> int:
        return len(self.thrusters) + len(self.flapped_foils)

    def to(self, device: str | torch.device) -> "Vessel":
        moved = replace(
            self,
            body=self.body.to(device),
            hull=self.hull.to(device),
            foils=[foil.to(device) for foil in self.foils],
            thrusters=[thruster.to(device) for thruster in self.thrusters],
        )
        moved.free_dof_mask = self.free_dof_mask.to(device)
        return moved


def gravity_force(mass: float, pose: Pose) -> Tensor:
    """Weight m g along NED z, expressed in the body frame [envs, 6]."""
    weight_body = mass * GRAVITY * pose.rotation[:, 2, :]
    return torch.cat((weight_body, torch.zeros_like(weight_body)), dim=-1)


def thrust_force(thrusters: list[Thruster], commands: Tensor) -> Tensor:
    """Thruster forces and moments in the body frame, commands [envs, num_thrusters]."""
    tau = torch.zeros(commands.shape[0], 6, device=commands.device)
    for i, thruster in enumerate(thrusters):
        force = commands[:, i, None] * thruster.force_vector
        moment = torch.linalg.cross(thruster.position_vector.expand_as(force), force)
        tau += torch.cat((force, moment), dim=-1)
    return tau


def flap_angles(vessel: Vessel, action: Tensor) -> list[Tensor]:
    """Flap deflection in rad for every foil, zero for foils without a flap."""
    flap_commands = iter(action[:, len(vessel.thrusters) :].unbind(-1))
    zero = torch.zeros(action.shape[0], device=action.device)
    return [next(flap_commands) * foil.max_flap if foil.has_flap else zero for foil in vessel.foils]


def water_around(vessel: Vessel, sea: Sea, t: Tensor, pose: Pose) -> list[Water]:
    """The sea at the hull samples, then at each foil's strips, from one evaluation."""
    point_sets = [vessel.hull.centers] + [foil.strip_centers for foil in vessel.foils]
    return water_at_point_sets(sea, t, pose, point_sets)


def water_at_point_sets(sea: Sea, t: Tensor, pose: Pose, point_sets: list[Tensor]) -> list[Water]:
    """The sea at several sets of body-frame points [P_i, 3], from one evaluation."""
    water = water_at(sea, pose.to_world(torch.cat(point_sets)), t)
    sizes = [len(points) for points in point_sets]
    return [Water(*parts) for parts in zip(*(x.split(sizes, dim=1) for x in water), strict=True)]


def all_foil_loads(
    vessel: Vessel,
    sea: Sea,
    t: Tensor,
    pose: Pose,
    nu: Tensor,
    action: Tensor,
    ventilated: Tensor,
    waters: list[Water] | None = None,
) -> list[FoilLoads]:
    """Loads on every foil; `ventilated` is [envs, num_foils] bool. `waters` is the output of
    `water_around` if the caller has it; otherwise the sea is evaluated at the foils only."""
    if not vessel.foils:
        return []
    flaps = flap_angles(vessel, action)
    if waters is None:
        foil_waters = water_at_point_sets(sea, t, pose, [f.strip_centers for f in vessel.foils])
    else:
        foil_waters = waters[1:]
    return [
        foil_loads(
            foil, sea, t, pose, nu, flaps[i], ventilated[:, i], vessel.density, foil_waters[i]
        )
        for i, foil in enumerate(vessel.foils)
    ]


def generalized_force(
    vessel: Vessel,
    sea: Sea,
    t: Tensor,
    pose: Pose,
    nu: Tensor,
    action: Tensor,
    ventilated: Tensor,
    waters: list[Water] | None = None,
) -> Tensor:
    """tau [envs, 6] in the body frame. `waters` (from `water_around`) skips the wave
    evaluation, for the frozen-wave approximation."""
    if waters is None:
        waters = water_around(vessel, sea, t, pose)
    tau = gravity_force(vessel.body.mass, pose)
    tau = tau + hydrostatic_force(vessel.hull, sea, t, pose, vessel.density, waters[0])
    tau = tau + thrust_force(vessel.thrusters, action[:, : len(vessel.thrusters)])
    loads = all_foil_loads(vessel, sea, t, pose, nu, action, ventilated, waters)
    for foil, foil_load in zip(vessel.foils, loads, strict=True):
        tau = tau + foil_wrench(foil, foil_load)
    return tau


def vessel_derivative(
    vessel: Vessel,
    sea: Sea,
    action: Tensor,
    ventilated: Tensor,
    waters: list[Water] | None = None,
) -> Derivative:
    """The right-hand side (eta_dot, nu_dot) with action and ventilation held over a step.

    With `waters`, the water elevation, velocity and acceleration at the hull and foil points
    stay frozen at those values instead of following t and the pose (an approximation).
    """

    def derivative(t: Tensor, eta: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
        pose = Pose.from_eta(eta)
        tau = generalized_force(vessel, sea, t, pose, nu, action, ventilated, waters)
        nu_dot = vessel.free_dof_mask * acceleration(vessel.body, nu, tau)
        return kinematics(eta, nu, pose), nu_dot

    return derivative
