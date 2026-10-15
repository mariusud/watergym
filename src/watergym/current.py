"""Ocean current: uniform in x and y, optionally varying with depth, one per environment.

The current flows horizontally at speed U f(d) toward the direction beta, measured from
north (x) toward east (y), where d is the depth below the mean surface and f a profile:

    v_c = U f(d) (cos(beta), sin(beta), 0)        in NED

Hull damping and added mass see the body velocity relative to the current at the CG
(rigid_body.acceleration). Foil strips add the current at their own depth to the flow.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from watergym.rigid_body import Pose


@dataclass
class Current:
    """`speed` [envs] in m/s and `direction` [envs] in rad, the way the water flows.

    An optional profile, shared by all envs, scales the speed with depth: `profile_factors`
    [D] at `profile_depths` [D] m below the mean surface, increasing, linear in between and
    constant beyond the ends. Points above the mean surface use the factor at depth 0.
    """

    speed: Tensor
    direction: Tensor
    profile_depths: Tensor | None = None
    profile_factors: Tensor | None = None

    @classmethod
    def uniform(
        cls,
        num_envs: int,
        speed: float | Tensor,
        direction: float | Tensor = 0.0,
        device: str | torch.device = "cpu",
    ) -> "Current":
        """The same current at every depth; scalars are repeated over the envs."""

        def per_env(value: float | Tensor) -> Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            return tensor.expand(num_envs).clone()

        return cls(per_env(speed), per_env(direction))

    def subset(self, index: slice | list[int] | Tensor) -> "Current":
        return Current(
            self.speed[index], self.direction[index], self.profile_depths, self.profile_factors
        )

    def replace(self, env_ids: Tensor, other: "Current") -> None:
        self.speed[env_ids] = other.speed
        self.direction[env_ids] = other.direction


def profile(current: Current, depth: Tensor) -> tuple[Tensor, Tensor]:
    """The profile factor f(d) and its slope df/dd at depths [envs, P], d = z clamped at 0."""
    if current.profile_depths is None or current.profile_factors is None:
        return torch.ones_like(depth), torch.zeros_like(depth)
    depths, factors = current.profile_depths, current.profile_factors
    clamped = depth.clamp(depths[0], depths[-1]).contiguous()
    upper = torch.searchsorted(depths, clamped, right=True).clamp(1, len(depths) - 1)
    d0, d1 = depths[upper - 1], depths[upper]
    f0, f1 = factors[upper - 1], factors[upper]
    slope = (f1 - f0) / (d1 - d0)
    inside = (depth > depths[0]) & (depth < depths[-1])
    return f0 + slope * (clamped - d0), torch.where(inside, slope, 0.0)


def current_velocity(current: Current, points: Tensor) -> Tensor:
    """v_c = U f(d) (cos(beta), sin(beta), 0) in NED at world points [envs, P, 3]."""
    factor, _ = profile(current, points[..., 2].clamp(min=0))
    speed = current.speed[:, None] * factor
    return torch.stack(
        (
            speed * torch.cos(current.direction)[:, None],
            speed * torch.sin(current.direction)[:, None],
            torch.zeros_like(speed),
        ),
        dim=-1,
    )


def body_current(current: Current, pose: Pose, nu: Tensor) -> tuple[Tensor, Tensor]:
    """nu_c and its time derivative at the CG, both [envs, 6] with zero angular part.

        v_c^b     = R^T v_c^n(z)
        v_c^b_dot = -omega x v_c^b + R^T (d v_c^n / dz) z_dot

    The first term is Fossen's for a current constant in NED (Handbook of Marine Craft
    Hydrodynamics and Motion Control, 2nd ed., 2021, sec. 10.3). The second follows the CG
    as it moves through a depth profile, and is zero for a uniform current.
    """
    flow_direction = torch.stack(
        (
            torch.cos(current.direction),
            torch.sin(current.direction),
            torch.zeros_like(current.direction),
        ),
        dim=-1,
    )
    factor, slope = profile(current, pose.position[:, 2:].clamp(min=0))
    rotation_transpose = pose.rotation.transpose(-1, -2)
    velocity_ned = current.speed[:, None] * factor * flow_direction
    velocity = (rotation_transpose @ velocity_ned[..., None]).squeeze(-1)
    omega = nu[:, 3:]
    rate = torch.linalg.cross(velocity, omega)
    if current.profile_depths is not None:
        heave_rate = (pose.rotation[:, 2, :] * nu[:, :3]).sum(-1, keepdim=True)
        shear_ned = current.speed[:, None] * slope * heave_rate * flow_direction
        rate = rate + (rotation_transpose @ shear_ned[..., None]).squeeze(-1)
    zeros = torch.zeros_like(velocity)
    return torch.cat((velocity, zeros), -1), torch.cat((rate, zeros), -1)
