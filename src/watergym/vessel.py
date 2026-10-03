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
from watergym.waves import GRAVITY, Sea

Vec6 = tuple[float, float, float, float, float, float]


@dataclass
class Thruster:
    position: Vec3
    direction: Vec3
    max_force_n: float


@dataclass
class Vessel:
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

    @property
    def flapped_foils(self) -> list[Foil]:
        return [foil for foil in self.foils if foil.has_flap]

    @property
    def num_actions(self) -> int:
        return len(self.thrusters) + len(self.flapped_foils)

    def to(self, device: str | torch.device) -> "Vessel":
        return replace(self, body=self.body.to(device), hull=self.hull.to(device))


def gravity_force(mass: float, pose: Pose) -> Tensor:
    """Weight m g along NED z, expressed in the body frame [envs, 6]."""
    weight_body = mass * GRAVITY * pose.rotation[:, 2, :]
    return torch.cat((weight_body, torch.zeros_like(weight_body)), dim=-1)


def thrust_force(thrusters: list[Thruster], commands: Tensor) -> Tensor:
    """Thruster forces and moments in the body frame, commands [envs, num_thrusters]."""
    tau = torch.zeros(commands.shape[0], 6, device=commands.device)
    for i, thruster in enumerate(thrusters):
        position = torch.tensor(thruster.position, device=commands.device)
        force = (
            commands[:, i, None]
            * thruster.max_force_n
            * torch.tensor(thruster.direction, device=commands.device)
        )
        tau += torch.cat((force, torch.linalg.cross(position.expand_as(force), force)), dim=-1)
    return tau


def flap_angles(vessel: Vessel, action: Tensor) -> list[Tensor]:
    """Flap deflection in rad for every foil, zero for foils without a flap."""
    flap_commands = iter(action[:, len(vessel.thrusters) :].unbind(-1))
    zero = torch.zeros(action.shape[0], device=action.device)
    return [next(flap_commands) * foil.max_flap if foil.has_flap else zero for foil in vessel.foils]


def all_foil_loads(
    vessel: Vessel, sea: Sea, t: Tensor, pose: Pose, nu: Tensor, action: Tensor, ventilated: Tensor
) -> list[FoilLoads]:
    """Loads on every foil; `ventilated` is [envs, num_foils] bool."""
    flaps = flap_angles(vessel, action)
    return [
        foil_loads(foil, sea, t, pose, nu, flaps[i], ventilated[:, i], vessel.density)
        for i, foil in enumerate(vessel.foils)
    ]


def generalized_force(
    vessel: Vessel, sea: Sea, t: Tensor, pose: Pose, nu: Tensor, action: Tensor, ventilated: Tensor
) -> Tensor:
    """tau [envs, 6] in the body frame."""
    tau = gravity_force(vessel.body.mass, pose)
    tau = tau + hydrostatic_force(vessel.hull, sea, t, pose, vessel.density)
    tau = tau + thrust_force(vessel.thrusters, action[:, : len(vessel.thrusters)])
    for foil, loads in zip(
        vessel.foils, all_foil_loads(vessel, sea, t, pose, nu, action, ventilated), strict=True
    ):
        tau = tau + foil_wrench(foil, loads)
    return tau


def vessel_derivative(vessel: Vessel, sea: Sea, action: Tensor, ventilated: Tensor) -> Derivative:
    """The right-hand side (eta_dot, nu_dot) with action and ventilation held over a step."""
    free = torch.tensor(vessel.free_dofs, dtype=torch.float32, device=action.device)

    def derivative(t: Tensor, eta: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
        pose = Pose.from_eta(eta)
        tau = generalized_force(vessel, sea, t, pose, nu, action, ventilated)
        return kinematics(eta, nu, pose), free * acceleration(vessel.body, nu, tau)

    return derivative
