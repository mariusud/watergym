"""Rare crashes of a controlled Moth in a Gaussian sea: limit state, design points (FORM, and a
gradient-free cross-entropy mean shift) and importance sampling.

The sea's random variables are standard normals z = (u, v, w, s), one of each per component:

    elevation_i = sigma_i (u_i cos(theta_i) - v_i sin(theta_i)),   sigma_i^2 = S(omega_i) d_omega
    omega_i     = bin centre + (Phi(w_i) - 1/2) bin width
    direction_i = heading + F^-1(Phi(s_i)),   F the cos-2s spreading distribution

Same spectrum, bins and spreading as `make_sea`, but `make_sea` fixes each amplitude at
sqrt(2 S d_omega) and draws only the phase. Here the amplitudes are Rayleigh, which makes the
elevation exactly Gaussian. The two seas have the same variance and different tails.

The limit state g(z) is a soft minimum over a RideControl-Moth-v1 window of three margins in
metres: hull clearance above the local surface, the main foil's distance from ventilation
onset, and the distance to the pitch limit. g <= 0 approximates the task's crash; the boolean
crash is returned too and is what importance sampling counts. Each margin also has its own
soft minimum over time, so FORM can search one failure mode at a time: the pitch margin sits
near 0.13 m in level flight and otherwise hides the other two from the overall minimum.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from watergym.env import SeaState, advance
from watergym.foils import foil_loads
from watergym.rigid_body import Pose
from watergym.tasks.ride_control import DT, MAX_PITCH, SAIL_COMMAND, RideControlMoth
from watergym.vessel import vessel_derivative
from watergym.vessels.moth_vessel import FLAP_RANGE
from watergym.waves import GRAVITY, Sea, Water, jonswap, water_at

Policy = Callable[[Tensor], Tensor]  # task observations [envs, 6] -> flap command [envs, 1]

MODES = ("hull", "foil", "pitch")
# Angles enter the margin as metres: 10 deg of angle-of-attack margin counts as 0.087 m.
ANGLE_SCALE_M = 0.5
MARGIN_TEMPERATURE_M = 0.005


@dataclass
class GaussianSea:
    """The deterministic part of a sea state; `sea(z)` turns standard normals into a `Sea`."""

    sigma: Tensor
    bin_center: Tensor
    bin_width: Tensor
    heading_rad: float
    direction_cdf: tuple[Tensor, Tensor] | None  # (offsets, cumulative probability)

    @classmethod
    def from_sea_state(cls, state: SeaState) -> "GaussianSea":
        n = state.num_components
        omega_p = 2 * math.pi / state.tp
        edges = torch.linspace(0.5, 4.0, n + 1) * omega_p
        width = edges[1:] - edges[:-1]
        center = (edges[:-1] + edges[1:]) / 2
        hs, tp = torch.tensor(state.hs), torch.tensor(state.tp)
        sigma = torch.sqrt(jonswap(center, hs, tp, state.gamma) * width)
        cdf = None
        if state.spreading is not None:
            offsets = torch.linspace(-math.pi, math.pi, 4001)
            density = torch.cos(offsets / 2).abs() ** (2 * state.spreading)
            cumulative = torch.cat((torch.zeros(1), torch.cumsum(density[1:] + density[:-1], 0)))
            cdf = (offsets, cumulative / cumulative[-1])
        return cls(sigma, center, width, state.heading_rad, cdf)

    @property
    def num_components(self) -> int:
        return len(self.sigma)

    @property
    def dim(self) -> int:
        return 4 * self.num_components

    def omega(self, z: Tensor) -> Tensor:
        w = z.unflatten(-1, (4, -1))[..., 2, :]
        return self.bin_center + (torch.special.ndtr(w) - 0.5) * self.bin_width

    def direction(self, z: Tensor) -> Tensor:
        s = z.unflatten(-1, (4, -1))[..., 3, :]
        direction = torch.full_like(s, self.heading_rad)
        if self.direction_cdf is not None:
            offsets, cumulative = self.direction_cdf
            direction = direction + interpolate(torch.special.ndtr(s), cumulative, offsets)
        return direction

    def encounter_frequency(self, z: Tensor, speed: float = 0.0) -> Tensor:
        """omega - k U cos(direction): how fast each phase turns at a point moving at U along +x."""
        omega = self.omega(z)
        return omega - omega**2 / GRAVITY * speed * torch.cos(self.direction(z))

    def sea(self, z: Tensor) -> Sea:
        """z [envs, 4 * num_components] -> one Sea per env."""
        u, v, _, _ = z.unflatten(-1, (4, -1)).unbind(-2)
        omega = self.omega(z)
        return Sea(
            amplitude=self.sigma * torch.sqrt(u**2 + v**2 + 1e-12),
            omega=omega,
            wavenumber=omega**2 / GRAVITY,
            direction=self.direction(z),
            phase=torch.atan2(v, u),
        )

    def time_shift(
        self, z: Tensor, delay_s: float | Tensor, distance_m: float | Tensor = 0.0
    ) -> Tensor:
        """The same sea delayed and moved along +x: sea(time_shift(z, d, D)) at (t, x) equals
        sea(z) at (t - d, x - D). With D the distance a vessel covers in d, it meets at t + d
        what it met at t.

        A rotation of each (u_i, v_i) by omega_i d - k_i cos(direction_i) D, so it maps
        N(0, I) onto itself."""
        u, v, w, s = z.unflatten(-1, (4, -1)).unbind(-2)
        omega = self.omega(z)
        delay = torch.as_tensor(delay_s).reshape(-1, 1)
        distance = torch.as_tensor(distance_m).reshape(-1, 1)
        angle = omega * delay - omega**2 / GRAVITY * torch.cos(self.direction(z)) * distance
        cos, sin = torch.cos(angle), torch.sin(angle)
        return torch.cat((u * cos - v * sin, u * sin + v * cos, w, s), -1)


def interpolate(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """np.interp for increasing xp, differentiable in x."""
    i = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0).clamp(min=1e-12)


@dataclass
class Rollout:
    """margin [envs] is the smooth limit state and mode_margin [envs, 3] the same per failure
    mode (see MODES); crash_mode is the index of the mode with the smallest hard margin at the
    crash step, -1 without a crash. Per step [envs, steps]: the hard margin, the wave
    elevation at the bow sensor, the flap command and the hull's x position;
    step_mode_margin [envs, steps, 3] is the hard margin per mode."""

    margin: Tensor
    mode_margin: Tensor
    crashed: Tensor
    crash_step: Tensor
    crash_mode: Tensor
    step_margin: Tensor
    step_mode_margin: Tensor
    bow_elevation: Tensor
    flap: Tensor
    position_x: Tensor


def soft_min(x: Tensor, dim: int | tuple[int, ...], temperature: float) -> Tensor:
    """-T logsumexp(-x / T): never above the hard minimum, at most T ln(n) below it."""
    return -temperature * torch.logsumexp(-x / temperature, dim=dim)


def rollout(
    gaussian_sea: GaussianSea,
    z: Tensor,
    policy: Policy,
    steps: int = 1000,
    checkpoint_steps: int = 0,
    temperature: float = MARGIN_TEMPERATURE_M,
) -> Rollout:
    """One RideControl-Moth-v1 window per row of z, from the task's initial state, no resets.

    The physics, observations and crash conditions are the task's. After its first crash an
    env's state is frozen, so later steps add nothing to its margin or its gradient.
    `checkpoint_steps` > 0 recomputes the forward pass in chunks of that many steps during
    backward, for memory.
    """
    num_envs = z.shape[0]
    task = RideControlMoth(1, device=z.device)
    vessel = task.env.vessel
    main_foil = vessel.foils[0]
    sea = gaussian_sea.sea(z)
    strips = main_foil.strip_centers
    points = torch.cat((task.points, strips))
    num_task_points = len(task.points)

    def measure(t: Tensor, eta: Tensor) -> tuple[Pose, Water]:
        pose = Pose.from_eta(eta)
        return pose, water_at(sea, pose.to_world(points), t)

    def heights_above_surface(pose: Pose, water: Water) -> Tensor:
        world_z = pose.to_world(task.points)[..., 2]
        return -(world_z + water.elevation[:, :num_task_points])

    def step(state: tuple[Tensor, ...]) -> tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        t, eta, nu, ventilated, wetting, last_flap, heave_accel, bow, dead = state
        obs = torch.stack((bow, eta[:, 4], nu[:, 4], heave_accel, nu[:, 0], last_flap), -1)
        flap = policy(obs).reshape(num_envs).clamp(-1, 1)
        action = torch.stack((torch.full_like(flap, SAIL_COMMAND), flap), -1)
        wetting1 = wetting.clone()  # advance updates it in place
        t1, eta1, nu1, vent1 = advance(
            vessel, sea, t, eta, nu, action, ventilated, wetting1, DT, 2, False
        )
        frozen = dead[:, None]
        wetting1 = torch.where(frozen, wetting, wetting1)
        t1 = torch.where(dead, t, t1)
        eta1, nu1 = torch.where(frozen, eta, eta1), torch.where(frozen, nu, nu1)
        vent1 = torch.where(frozen, ventilated, vent1)
        flap = torch.where(dead, last_flap, flap)

        _, nu_dot = vessel_derivative(vessel, sea, action, vent1)(t1, eta1, nu1)
        pose, water = measure(t1, eta1)
        linear = nu_dot[:, :3] + torch.linalg.cross(nu1[:, 3:], nu1[:, :3])
        accel1 = -(pose.rotation[:, 2] * linear).sum(-1)
        heights = heights_above_surface(pose, water)
        strip_water = Water(*(x[:, num_task_points:] for x in water))
        loads = foil_loads(
            main_foil, sea, t1, pose, nu1, flap * FLAP_RANGE, vent1[:, 0], water=strip_water
        )

        pitch = eta1[:, 4]
        diverged = ~(torch.isfinite(eta1).all(-1) & torch.isfinite(nu1).all(-1))
        touchdown = heights[:, 3:].amin(-1) < 0
        crashed = touchdown | vent1[:, 0] | (pitch.abs() > MAX_PITCH) | diverged

        chord = main_foil.chord_m
        to_onset = torch.stack(
            (
                loads.depth - main_foil.onset_depth_chords * chord,
                ANGLE_SCALE_M * (main_foil.ventilation_onset_angle - loads.alpha.abs()),
                -(loads.depth + main_foil.immersion_band / 2),
            )
        ).amax(0)
        terms = torch.stack(
            (
                heights[:, 3:].amin(-1),
                to_onset.amin(-1),
                ANGLE_SCALE_M * (MAX_PITCH - pitch.abs()),
            ),
            -1,
        )
        terms = torch.where(torch.isfinite(terms), terms, -1.0)
        terms = torch.where(dead[:, None], 1.0, terms)
        elevation = water.elevation[:, 0].detach()
        next_state = (
            t1,
            eta1,
            nu1,
            vent1,
            wetting1,
            flap,
            torch.where(dead, heave_accel, accel1),
            torch.where(dead, bow, heights[:, 0]),
            dead | crashed,
        )
        extra = torch.stack((elevation, flap.detach(), eta1[:, 0].detach()), -1)
        return next_state, terms, crashed & ~dead, extra

    def chunk(n: int, *state: Tensor) -> tuple[Tensor, ...]:
        records = []
        for _ in range(n):
            state, terms, new_crash, extra = step(state)
            records.append((terms, new_crash, extra))
        terms, new_crash, extra = (torch.stack(r, 1) for r in zip(*records, strict=True))
        return (*state, terms, new_crash, extra)

    eta = torch.tensor(vessel.initial_eta, device=z.device).expand(num_envs, 6).clone()
    nu = torch.tensor(vessel.initial_nu, device=z.device).expand(num_envs, 6).clone()
    t = torch.zeros(num_envs, device=z.device)
    pose, water = measure(t, eta)
    zeros = torch.zeros(num_envs, device=z.device)
    no_foils = torch.zeros(num_envs, len(vessel.foils), dtype=torch.bool, device=z.device)
    state = (
        t,
        eta,
        nu,
        no_foils,
        torch.zeros(no_foils.shape, device=z.device),
        zeros,
        zeros,
        heights_above_surface(pose, water)[:, 0],
        torch.zeros(num_envs, dtype=torch.bool, device=z.device),
    )

    outputs = []
    size = checkpoint_steps or steps
    for start in range(0, steps, size):
        n = min(size, steps - start)
        if checkpoint_steps and torch.is_grad_enabled():
            result = checkpoint(chunk, n, *state, use_reentrant=False)
        else:
            result = chunk(n, *state)
        state, recorded = result[:-3], result[-3:]
        outputs.append(recorded)
    terms, new_crash, extra = (torch.cat(o, 1) for o in zip(*outputs, strict=True))

    crashed = new_crash.any(1)
    first = new_crash.float().argmax(1)
    at_crash = terms.detach()[torch.arange(num_envs, device=z.device), first]
    return Rollout(
        margin=soft_min(terms, (1, 2), temperature),
        mode_margin=soft_min(terms, 1, temperature),
        crashed=crashed,
        crash_step=torch.where(crashed, first, -1),
        crash_mode=torch.where(crashed, at_crash.argmin(-1), -1),
        step_margin=terms.amin(-1).detach(),
        step_mode_margin=terms.detach(),
        bow_elevation=extra[..., 0],
        flap=extra[..., 1],
        position_x=extra[..., 2],
    )


@dataclass
class DesignPoints:
    """FORM results per start: z [starts, dim], beta = |z|, the limit state at z, and whether
    the boolean crash happens there."""

    z: Tensor
    beta: Tensor
    margin: Tensor
    failed: Tensor
    history: list[dict]


def form(
    limit_state: Callable[[Tensor], tuple[Tensor, Tensor]],
    z0: Tensor,
    iterations: int = 100,
    lr: float = 0.05,
    penalty: float = 1.0,
    state: dict | None = None,
    callback: Callable[[int, dict], None] | None = None,
) -> DesignPoints:
    """The design point nearest the origin on g(z) = 0, from every row of z0.

    Adam on the augmented Lagrangian 1/2 |z|^2 + lam g + rho/2 (g / |grad g|)^2, with the
    least-squares multiplier lam = -z.grad g / |grad g|^2 at every step. Its gradient is the
    part of z along the surface plus rho times the distance to the surface, so it shrinks |z|
    while it walks onto g = 0, from either side. A multiplier updated by lam += rho g
    oscillates under Adam's bounded steps. The learning rate anneals to a tenth by a cosine.
    `limit_state(z)` returns (g, failed). The result per start is its smallest-|z| iterate
    that failed, or its last iterate (failed = False) if none did.

    Pass the same `state` dict to resume: it holds everything needed to continue.
    """
    if state is None or not state:
        state = state if state is not None else {}
        state |= {
            "z": z0.clone(),
            "adam": None,
            "iteration": 0,
            "best_z": z0.clone(),
            "best_beta": torch.full((len(z0),), math.inf),
            "best_margin": torch.full((len(z0),), math.nan),
            "history": [],
        }
    z = state["z"].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    if state["adam"] is not None:
        optimizer.load_state_dict(state["adam"])
    best_z, best_beta, best_margin = state["best_z"], state["best_beta"], state["best_margin"]
    while state["iteration"] < iterations:
        i = state["iteration"]
        for group in optimizer.param_groups:
            group["lr"] = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * i / iterations)))
        g, failed = limit_state(z)
        beta = z.detach().norm(dim=-1)
        improved = failed & (beta < best_beta)
        best_z[improved], best_beta[improved] = z.detach()[improved], beta[improved]
        best_margin[improved] = g.detach()[improved]
        (grad,) = torch.autograd.grad(g.sum(), z)
        grad_norm2 = (grad**2).sum(-1, keepdim=True).clamp(min=1e-12)
        zd = z.detach()
        lam = -(zd * grad).sum(-1, keepdim=True) / grad_norm2
        z.grad = zd + lam * grad + penalty * g.detach()[:, None] * grad / grad_norm2
        optimizer.step()
        record = {"iteration": i, "beta": beta, "margin": g.detach(), "failed": failed}
        state["history"].append(record)
        state |= {"z": z.detach().clone(), "adam": optimizer.state_dict()}
        state["iteration"] = i + 1
        if callback:
            callback(i, record)
    never = torch.isinf(best_beta)
    best_z[never] = state["z"][never]
    best_beta[never] = best_z[never].norm(dim=-1)
    best_margin[never] = state["history"][-1]["margin"][never]
    return DesignPoints(best_z, best_beta, best_margin, ~never, state["history"])


@dataclass
class CrossEntropyShift:
    """The mean of the last sampling density, and per level the threshold and the means."""

    mean: Tensor
    thresholds: list[float]
    means: list[Tensor]


def cross_entropy_shift(
    score: Callable[[Tensor], tuple[Tensor, Tensor]],
    dim: int,
    samples: int = 1024,
    quantile: float = 0.2,
    max_levels: int = 20,
    generator: torch.Generator | None = None,
    state: dict | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> CrossEntropyShift:
    """A mean shift towards {score(z) <= 0} for z ~ N(0, I), without gradients.

    Each level samples N(mean, I), takes the `quantile` of the scores, or 0 once that many
    samples fail, as the level's threshold, and moves the mean to the plain average of the
    samples at or below it. It stops at the level whose threshold is 0, so the last mean is
    the average failure under the last density: a little beyond the design point, a good
    centre for importance sampling. Likelihood-ratio weights would make it E[z | failure],
    but they degenerate in a few hundred dimensions once |mean| passes about 2.

    `score(z)` returns the scores [n], failure at <= 0, and the samples to average, which may
    be mapped by any transform that keeps N(0, I), such as a time shift that puts every
    sample's worst moment at the same time. Without that, a stationary sea's failures at
    different times average out. Pass the same `state` dict to resume.
    """
    generator = generator or torch.Generator().manual_seed(0)
    if state is None or not state:
        state = state if state is not None else {}
        state |= {"mean": torch.zeros(dim, dtype=torch.float64), "thresholds": [], "means": []}
        state["seed"] = generator.initial_seed()
    while len(state["thresholds"]) < max_levels and (
        not state["thresholds"] or state["thresholds"][-1] > 0
    ):
        level = len(state["thresholds"])
        generator.manual_seed(state["seed"] + level)
        z = state["mean"] + torch.randn(samples, dim, generator=generator, dtype=torch.float64)
        s, mapped = score(z.float())
        s = s.double().cpu()
        threshold = max(torch.quantile(s, quantile).item(), 0.0)
        state["mean"] = mapped.double().cpu()[s <= threshold].mean(0)
        state["thresholds"].append(threshold)
        state["means"].append(state["mean"])
        if callback:
            callback(level, threshold)
    return CrossEntropyShift(state["mean"], state["thresholds"], state["means"])


def normal_cdf(x: float | Tensor) -> Tensor:
    return torch.special.ndtr(torch.as_tensor(x, dtype=torch.float64))


def outcrossing_rate(gaussian_sea: GaussianSea, z: Tensor, speed: float = 0.0) -> Tensor:
    """Rice's mean upcrossing rate [1/s] of the linearised limit state at a design point z,
    for a stationary sea and a vessel at constant speed along +x: |d alpha / dt|
    exp(-beta^2 / 2) / (2 pi), alpha = z / beta. The design point moves in time by rotating
    each (u_i, v_i) at its encounter frequency (see `time_shift`)."""
    u, v, _, _ = z.unflatten(-1, (4, -1)).unbind(-2)
    beta = z.norm(dim=-1)
    rate = gaussian_sea.encounter_frequency(z, speed)
    alpha_rate = torch.sqrt((rate**2 * (u**2 + v**2)).sum(-1)) / beta
    return alpha_rate * torch.exp(-(beta.double() ** 2) / 2) / (2 * math.pi)


@dataclass
class ImportanceEstimate:
    probability: float
    standard_error: float
    samples: int
    hits: int
    effective_samples: float


def importance_sampling(
    failed: Callable[[Tensor], Tensor],
    centers: Tensor,
    samples: int,
    batch: int = 1024,
    generator: torch.Generator | None = None,
) -> ImportanceEstimate:
    """P(failed(z)) for z ~ N(0, I), sampling from an equal mixture of N(center_k, I).

    The weight phi(z) / q(z) uses the full mixture density, so the estimate is unbiased
    whatever the centres are; centres near the failure region only lower its variance.
    """
    generator = generator or torch.Generator().manual_seed(0)
    centers = centers.double()
    log_k = math.log(len(centers))
    weights, hits = [], []
    for start in range(0, samples, batch):
        n = min(batch, samples - start)
        pick = torch.randint(len(centers), (n,), generator=generator)
        z = centers[pick] + torch.randn(n, centers.shape[1], generator=generator).double()
        log_q = torch.logsumexp(-0.5 * torch.cdist(z, centers) ** 2, 1) - log_k
        weights.append(torch.exp(-0.5 * (z**2).sum(-1) - log_q))
        hits.append(failed(z.float()).cpu())
    w, hit = torch.cat(weights), torch.cat(hits).double()
    terms = w * hit
    p = terms.mean().item()
    se = terms.std().item() / math.sqrt(len(terms))
    ess = (terms.sum() ** 2 / (terms**2).sum()).item() if hit.any() else 0.0
    return ImportanceEstimate(p, se, len(terms), int(hit.sum()), ess)
