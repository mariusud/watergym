"""6-DOF rigid-body dynamics in Fossen's notation (Handbook of Marine Craft Hydrodynamics
and Motion Control, 2nd ed., 2021).

eta = [x, y, z, phi, theta, psi]: NED position and zyx Euler angles.
nu  = [u, v, w, p, q, r]: body-frame linear and angular velocity (x forward, y starboard,
z down). The body origin is the centre of gravity.

    eta_dot = J(eta) nu                                  (kinematics, Fossen ch. 2)
    (M_RB + M_A) nu_dot + C(nu) nu + D(nu) nu = tau      (kinetics, Fossen ch. 3 and 6)

Gravity, buoyancy, foils and thrusters all arrive through tau.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class RigidBody:
    mass: float
    inertia: Tensor
    added_mass: Tensor
    linear_damping: Tensor
    quadratic_damping: Tensor
    mass_matrix: Tensor = field(init=False)
    inverse_mass_matrix: Tensor = field(init=False)

    def __post_init__(self) -> None:
        """M = M_RB + M_A for a body whose origin is its centre of gravity."""
        rigid = torch.zeros_like(self.added_mass)
        rigid[:3, :3] = self.mass * torch.eye(3, device=rigid.device)
        rigid[3:, 3:] = self.inertia
        self.mass_matrix = rigid + self.added_mass
        self.inverse_mass_matrix = torch.linalg.inv(self.mass_matrix)

    @classmethod
    def from_diagonals(
        cls,
        mass: float,
        inertia: list[float],
        added_mass: list[float],
        linear_damping: list[float],
        quadratic_damping: list[float],
    ) -> "RigidBody":
        return cls(
            mass,
            torch.diag(torch.tensor(inertia)),
            torch.diag(torch.tensor(added_mass)),
            torch.tensor(linear_damping),
            torch.tensor(quadratic_damping),
        )

    def to(self, device: str | torch.device) -> "RigidBody":
        return RigidBody(
            self.mass,
            self.inertia.to(device),
            self.added_mass.to(device),
            self.linear_damping.to(device),
            self.quadratic_damping.to(device),
        )


@dataclass
class Pose:
    """Where a batch of bodies is: NED position [envs, 3] and rotation body to NED [envs, 3, 3]."""

    position: Tensor
    rotation: Tensor

    @classmethod
    def from_eta(cls, eta: Tensor) -> "Pose":
        return cls(eta[:, :3], rotation_matrix(eta[:, 3], eta[:, 4], eta[:, 5]))

    def to_world(self, points_body: Tensor) -> Tensor:
        """Body-frame points [P, 3] to NED positions [envs, P, 3].

        einsum runs this as one [P, 3] x [3, envs * 3] matrix product, about 25x faster on
        CPU at 1024 envs than broadcasting `points_body @ rotation^T` into a batched matmul.
        Its result is strided, and the wave math downstream runs 2x slower on MPS unless it
        is made contiguous.
        """
        rotated = torch.einsum("pj,eij->epi", points_body, self.rotation).contiguous()
        return self.position[:, None] + rotated

    def to_body(self, vectors_world: Tensor) -> Tensor:
        """NED vectors [envs, P, 3] to body-frame vectors, R^T v."""
        return vectors_world @ self.rotation


def rotation_matrix(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    """R_zyx(phi, theta, psi): body to NED, shaped [..., 3, 3]."""
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)
    return torch.stack(
        (
            torch.stack(
                (cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth), -1
            ),
            torch.stack(
                (spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi), -1
            ),
            torch.stack((-sth, cth * sphi, cth * cphi), -1),
        ),
        dim=-2,
    )


def euler_rate_matrix(phi: Tensor, theta: Tensor) -> Tensor:
    """T_zyx(phi, theta) mapping [p, q, r] to Euler angle rates. Singular at
    theta = +-90 deg, which a vessel never reaches."""
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, tth = torch.cos(theta), torch.tan(theta)
    one, zero = torch.ones_like(phi), torch.zeros_like(phi)
    return torch.stack(
        (
            torch.stack((one, sphi * tth, cphi * tth), -1),
            torch.stack((zero, cphi, -sphi), -1),
            torch.stack((zero, sphi / cth, cphi / cth), -1),
        ),
        dim=-2,
    )


def kinematics(eta: Tensor, nu: Tensor, pose: Pose) -> Tensor:
    """eta_dot = J(eta) nu."""
    position_rate = (pose.rotation @ nu[:, :3, None]).squeeze(-1)
    angle_rate = (euler_rate_matrix(eta[:, 3], eta[:, 4]) @ nu[:, 3:, None]).squeeze(-1)
    return torch.cat((position_rate, angle_rate), dim=-1)


def coriolis_force(mass_matrix: Tensor, nu: Tensor) -> Tensor:
    """C(nu) nu, with C from Fossen's m2c, written as cross products of the momenta
    p = M nu: [w x p_linear, v x p_linear + w x p_angular]."""
    momentum = nu @ mass_matrix.T
    v, w = nu[:, :3], nu[:, 3:]
    p_linear, p_angular = momentum[:, :3], momentum[:, 3:]
    cross = torch.linalg.cross
    return torch.cat((cross(w, p_linear), cross(v, p_linear) + cross(w, p_angular)), dim=-1)


def damping_force(body: RigidBody, nu: Tensor) -> Tensor:
    """D(nu) nu with diagonal linear and quadratic (|nu| nu) terms."""
    return body.linear_damping * nu + body.quadratic_damping * nu.abs() * nu


def acceleration(body: RigidBody, nu: Tensor, tau: Tensor) -> Tensor:
    """nu_dot = M^-1 (tau - C(nu) nu - D(nu) nu)."""
    net = tau - coriolis_force(body.mass_matrix, nu) - damping_force(body, nu)
    return net @ body.inverse_mass_matrix.T


Derivative = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


def rk4_step(
    derivative: Derivative, t: Tensor, eta: Tensor, nu: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """Classic fourth-order Runge-Kutta step for (eta, nu); derivative(t, eta, nu) returns
    (eta_dot, nu_dot)."""
    k1 = derivative(t, eta, nu)
    k2 = derivative(t + dt / 2, eta + dt / 2 * k1[0], nu + dt / 2 * k1[1])
    k3 = derivative(t + dt / 2, eta + dt / 2 * k2[0], nu + dt / 2 * k2[1])
    k4 = derivative(t + dt, eta + dt * k3[0], nu + dt * k3[1])
    eta_next = eta + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    nu_next = nu + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return eta_next, nu_next
