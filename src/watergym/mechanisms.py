"""Mechanical ride-height controllers for the Moth: a wand that is a real pendulum, and a
buildable linkage from wand to flap.

`wand_flap` in moth_vessel reads the water under the pivot exactly, so it cannot skip. Here
the wand is a rigid rod hinged at the bow, raked forward and down:

    I phi'' = gravity + bungee + tip contact + pivot acceleration + hinge friction

phi is the wand's angle below the world horizontal, positive with the tip down. Gravity and
the bungee swing the tip down; the water pushes it up with a one-sided penalty force
N = max(0, k p + c p') when the tip is p below the local surface, plus a planing drag mu N
aft. When the surface falls away faster than gravity and the bungee can follow, N is zero
and the wand skips.

The linkage turns the wand angle relative to the hull into a flap angle through a monotone
cam (a 5-knot shape-preserving cubic spline) and a first-order dashpot. `MechanicalWand` is
a stateful controller beside a RideControl-Moth-v1 task: it steps the wand between env steps
and returns the flap command, so the task's dynamics stay untouched. Every parameter is a
per-env tensor, so one batch can score many linkages at once.
"""

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from watergym.env import WaterEnv
from watergym.rigid_body import Pose
from watergym.tasks.ride_control import RideControlMoth
from watergym.vessels.moth_vessel import (
    FLAP_RANGE,
    WAND_GEARING,
    WAND_LENGTH_M,
    WAND_NEUTRAL_ANGLE,
    WAND_PIVOT,
)
from watergym.waves import GRAVITY, elevation

# Wand: carbon tube plus a planing paddle at the tip.
ROD_MASS_PER_M = 0.12
TIP_MASS_KG = 0.05
HINGE_DAMPING = 0.01  # N m s/rad
BUNGEE_STIFFNESS = 0.5  # N m/rad about WAND_NEUTRAL_ANGLE, on top of the preload
STOCK_BUNGEE_NM = 3.0
# Tip contact, estimated for a 5 x 10 cm paddle planing at 8.4 m/s: lift per unit immersion
# 1/2 rho U^2 b C_L and per unit heave rate 1/2 rho U S C_L_alpha. The wand is overdamped on
# the water, so it does not bounce off it.
CONTACT_STIFFNESS = 3000.0  # N/m
CONTACT_DAMPING = 120.0  # N s/m
PLANING_DRAG = 0.2  # drag / normal force
# Hinge stops, relative to the hull.
MIN_WAND_ANGLE, MAX_WAND_ANGLE = 0.0, math.radians(90)
SUBSTEPS = 25

# The cam's knots: wand angles relative to the hull. Between them a shape-preserving cubic,
# beyond them the end values.
CAM_KNOTS = torch.linspace(math.radians(5), math.radians(85), 5)


@dataclass
class Linkage:
    """A buildable wand linkage, one per env. Lengths in m, angles in rad, time in s."""

    cam_flap: Tensor  # [envs, 5] flap angle at each CAM_KNOTS wand angle, non-increasing
    wand_length_m: Tensor  # [envs]
    pivot_x_m: Tensor  # [envs], body frame, forward of the CG
    dashpot_s: Tensor  # [envs], first-order lag from cam to flap; 0 is rigid
    bungee_nm: Tensor  # [envs], preload torque swinging the tip down

    @property
    def num_envs(self) -> int:
        return self.cam_flap.shape[0]

    def to(self, device: str | torch.device) -> "Linkage":
        return Linkage(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def repeat_interleave(self, repeats: int) -> "Linkage":
        return Linkage(
            **{f.name: getattr(self, f.name).repeat_interleave(repeats, 0) for f in fields(self)}
        )


def stock_linkage(num_envs: int = 1) -> Linkage:
    """moth_vessel's wand: linear gearing, 1 m long at the bow, no dashpot."""
    cam = WAND_GEARING * (WAND_NEUTRAL_ANGLE - CAM_KNOTS)
    return Linkage(
        cam_flap=cam.expand(num_envs, -1).clone(),
        wand_length_m=torch.full((num_envs,), WAND_LENGTH_M),
        pivot_x_m=torch.full((num_envs,), WAND_PIVOT[0]),
        dashpot_s=torch.zeros(num_envs),
        bungee_nm=torch.full((num_envs,), STOCK_BUNGEE_NM),
    )


def cam(cam_flap: Tensor, wand_angle: Tensor) -> Tensor:
    """Flap angle [envs] from wand angle [envs] through the Fritsch-Carlson monotone cubic.

    On equally spaced knots the interior slopes are the harmonic mean of the neighbouring
    secant slopes (zero where they change sign or one is flat), the end slopes the end
    secants. That keeps the curve monotone, and exactly linear when the knots are.
    """
    knots = CAM_KNOTS.to(wand_angle.device)
    h = knots[1] - knots[0]
    secant = cam_flap.diff(dim=-1) / h
    left, right = secant[:, :-1], secant[:, 1:]
    same_sign = left * right > 0
    harmonic = 2 * left * right / torch.where(same_sign, left + right, torch.ones_like(left))
    interior = torch.where(same_sign, harmonic, torch.zeros_like(left))
    slope = torch.cat((secant[:, :1], interior, secant[:, -1:]), dim=-1)

    x = wand_angle.clamp(knots[0], knots[-1])
    i = ((x - knots[0]) / h).floor().long().clamp(0, len(knots) - 2)[:, None]
    s = ((x - knots[0]) / h - i[:, 0])[:, None]
    y0, y1 = cam_flap.gather(1, i), cam_flap.gather(1, i + 1)
    m0, m1 = slope.gather(1, i) * h, slope.gather(1, i + 1) * h
    s2, s3 = s * s, s * s * s
    return (
        (2 * s3 - 3 * s2 + 1) * y0
        + (s3 - 2 * s2 + s) * m0
        + (-2 * s3 + 3 * s2) * y1
        + (s3 - s2) * m1
    )[:, 0]


def hermite(
    p0: Tensor, v0: Tensor, p1: Tensor, v1: Tensor, s: float, dt: float
) -> tuple[Tensor, Tensor]:
    """Position and acceleration at fraction s of a step of a cubic through both ends'
    positions and velocities."""
    d0, d1 = v0 * dt, v1 * dt
    position = (
        (2 * s**3 - 3 * s**2 + 1) * p0
        + (s**3 - 2 * s**2 + s) * d0
        + (-2 * s**3 + 3 * s**2) * p1
        + (s**3 - s**2) * d1
    )
    accel = ((12 * s - 6) * p0 + (6 * s - 4) * d0 + (6 - 12 * s) * p1 + (6 * s - 2) * d1) / dt**2
    return position, accel


class MechanicalWand:
    """A per-env wand pendulum and linkage. Call it with a RideControl-Moth-v1 task before
    each `task.step` to get the flap command [envs, 1] in [-1, 1].

    Each call first advances the wand from the previous call's time to now, interpolating the
    pivot's motion between the two boat states, then reads the cam. Envs whose episode just
    started get the wand at rest on the water.
    """

    def __init__(self, linkage: Linkage, substeps: int = SUBSTEPS) -> None:
        self.linkage = linkage
        self.substeps = substeps
        self.length = linkage.wand_length_m
        rod = ROD_MASS_PER_M * self.length
        self.inertia = rod * self.length**2 / 3 + TIP_MASS_KG * self.length**2
        self.mass_moment = rod * self.length / 2 + TIP_MASS_KG * self.length
        self.angle: Tensor | None = None  # phi, below the world horizontal
        self.rate: Tensor | None = None
        self.in_contact: Tensor | None = None

    def __call__(self, task: RideControlMoth) -> Tensor:
        env = task.env
        pivot, velocity = self._pivot(env.eta, env.nu)
        pitch = env.eta[:, 4]
        if self.angle is None:
            self._allocate(env.num_envs, env.device)
        new = env.steps == 0
        if (~new).any():
            self._advance(env, pivot, velocity, pitch, ~new)
        if new.any():
            self._rest_on_water(env, pivot, pitch, new)
        self._previous = (pivot, velocity, pitch, env.t.clone())
        return (self.flap / FLAP_RANGE).clamp(-1, 1)[:, None]

    @property
    def relative_angle(self) -> Tensor:
        """Wand angle relative to the hull, what the linkage reads."""
        return self.angle + self._previous[2]

    def _allocate(self, num_envs: int, device: torch.device) -> None:
        self.linkage = self.linkage.to(device)
        for name in ("length", "inertia", "mass_moment"):
            setattr(self, name, getattr(self, name).to(device))
        zeros = torch.zeros(num_envs, device=device)
        self.angle, self.rate, self.flap, self.immersion = (zeros.clone() for _ in range(4))
        self.in_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _pivot(self, eta: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
        """World position and velocity of each env's pivot [envs, 3]."""
        body = torch.zeros(eta.shape[0], 3, device=eta.device)
        body[:, 0] = self.linkage.pivot_x_m.to(eta.device)
        body[:, 2] = WAND_PIVOT[2]
        rotation = Pose.from_eta(eta).rotation
        position = eta[:, :3] + torch.einsum("eij,ej->ei", rotation, body)
        velocity = torch.einsum(
            "eij,ej->ei", rotation, nu[:, :3] + torch.linalg.cross(nu[:, 3:], body)
        )
        return position, velocity

    def _tip(self, pivot: Tensor, angle: Tensor) -> Tensor:
        tip = pivot.clone()
        tip[:, 0] += self.length * torch.cos(angle)
        tip[:, 2] += self.length * torch.sin(angle)
        return tip

    def _rest_on_water(self, env: WaterEnv, pivot: Tensor, pitch: Tensor, ids: Tensor) -> None:
        height = -(pivot[:, 2] + elevation(env.sea, pivot[:, None], env.t)[:, 0])
        angle = torch.asin((height / self.length).clamp(0, 1))
        relative = (angle + pitch).clamp(MIN_WAND_ANGLE, MAX_WAND_ANGLE)
        self.angle[ids] = (relative - pitch)[ids]
        self.rate[ids] = 0.0
        tip = self._tip(pivot, self.angle)
        depth = tip[:, 2] + elevation(env.sea, tip[:, None], env.t)[:, 0]
        self.immersion[ids] = depth[ids]
        self.flap[ids] = cam(self.linkage.cam_flap, relative)[ids]

    def _advance(
        self, env: WaterEnv, pivot: Tensor, velocity: Tensor, pitch: Tensor, ids: Tensor
    ) -> None:
        pivot0, velocity0, pitch0, t0 = self._previous
        dt = env.dt
        h = dt / self.substeps
        lag = torch.exp(-h / self.linkage.dashpot_s.clamp(min=1e-6))
        angle, rate, immersion, flap = self.angle, self.rate, self.immersion, self.flap
        for k in range(1, self.substeps + 1):
            s = k / self.substeps
            position, accel = hermite(pivot0, velocity0, pivot, velocity, s, dt)
            hull_pitch = pitch0 + s * (pitch - pitch0)
            relative = angle + hull_pitch
            cos, sin = torch.cos(angle), torch.sin(angle)

            tip = self._tip(position, angle)
            depth = tip[:, 2] + elevation(env.sea, tip[:, None], t0 + s * dt)[:, 0]
            depth_rate = (depth - immersion) / h
            normal = (CONTACT_STIFFNESS * depth + CONTACT_DAMPING * depth_rate).clamp(min=0)
            normal = torch.where(depth > 0, normal, torch.zeros_like(normal))
            immersion = depth

            torque = (
                self.mass_moment * (GRAVITY - accel[:, 2]) * cos
                + self.mass_moment * accel[:, 0] * sin
                + self.linkage.bungee_nm
                - BUNGEE_STIFFNESS * (relative - WAND_NEUTRAL_ANGLE)
                - normal * self.length * (cos + PLANING_DRAG * sin)
                - HINGE_DAMPING * rate
            )
            rate = rate + h * torque / self.inertia
            angle = angle + h * rate
            relative = angle + hull_pitch
            at_stop = (relative < MIN_WAND_ANGLE) | (relative > MAX_WAND_ANGLE)
            angle = relative.clamp(MIN_WAND_ANGLE, MAX_WAND_ANGLE) - hull_pitch
            rate = torch.where(at_stop, torch.zeros_like(rate), rate)
            target = cam(self.linkage.cam_flap, relative.clamp(MIN_WAND_ANGLE, MAX_WAND_ANGLE))
            flap = target + (flap - target) * lag

        self.angle = torch.where(ids, angle, self.angle)
        self.rate = torch.where(ids, rate, self.rate)
        self.immersion = torch.where(ids, immersion, self.immersion)
        self.flap = torch.where(ids, flap.clamp(-FLAP_RANGE, FLAP_RANGE), self.flap)
        self.in_contact = torch.where(ids, normal > 0, self.in_contact)
