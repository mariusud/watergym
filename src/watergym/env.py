"""A batch of identical vessels, each in its own random sea, stepped together in PyTorch.

The API follows Gymnasium's vector envs: reset() -> (obs, info) and
step(action) -> (obs, reward, terminated, truncated, info), all tensors shaped [num_envs, ...].
Finished envs reset inside step(); their last observation is in info["final_obs"].
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from watergym.foils import FoilLoads, update_ventilation
from watergym.rigid_body import Pose, kinematics, rk4_step
from watergym.vessel import Vessel, all_foil_loads, vessel_derivative, water_around
from watergym.waves import Sea, make_sea

RewardFn = Callable[["WaterEnv"], Tensor]  # also the type of termination_fn: a bool [envs]


@dataclass
class SeaState:
    hs: float = 0.0
    tp: float = 6.0
    heading_rad: float = 0.0
    spreading: float | None = None
    gamma: float = 3.3
    num_components: int = 64


def advance(
    vessel: Vessel,
    sea: Sea,
    t: Tensor,
    eta: Tensor,
    nu: Tensor,
    action: Tensor,
    ventilated: Tensor,
    wetting_time: Tensor,
    dt: float,
    substeps: int,
    frozen_waves: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The physics of one env step: `substeps` RK4 steps, then the ventilation update.

    Returns the new (t, eta, nu, ventilated) and advances `wetting_time` in place. With
    `frozen_waves` the sea is evaluated once, at mid-step (t + dt/2, at the pose the current
    velocities reach by then), and reused by every RK4 stage. Ventilation still sees the sea
    at the end of the step: its thresholds flip on small depth errors.
    """
    waters = None
    if frozen_waves:
        midpoint = eta + dt / 2 * kinematics(eta, nu, Pose.from_eta(eta))
        waters = water_around(vessel, sea, t + dt / 2, Pose.from_eta(midpoint))
    derivative = vessel_derivative(vessel, sea, action, ventilated, waters)
    substep_dt = dt / substeps
    for _ in range(substeps):
        eta, nu = rk4_step(derivative, t, eta, nu, substep_dt)
        t = t + substep_dt

    loads = all_foil_loads(vessel, sea, t, Pose.from_eta(eta), nu, action, ventilated)
    next_ventilated = ventilated.clone()
    for i, foil in enumerate(vessel.foils):
        next_ventilated[:, i] = update_ventilation(
            foil, ventilated[:, i], loads[i], wetting_time[:, i], dt
        )
    return t, eta, nu, next_ventilated


def no_reward(env: "WaterEnv") -> Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def never_terminate(env: "WaterEnv") -> Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class WaterEnv:
    def __init__(
        self,
        vessel: Vessel,
        num_envs: int,
        sea_state: SeaState | None = None,
        dt: float = 0.02,
        substeps: int = 2,
        episode_length_s: float = 30.0,
        max_tilt: float = math.radians(60),
        reward_fn: RewardFn = no_reward,
        termination_fn: RewardFn = never_terminate,
        device: str | torch.device = "cpu",
        frozen_waves: bool = False,
        compile: bool = False,
    ) -> None:
        """`frozen_waves` and `compile` trade accuracy or start-up time for speed; both are
        off by default. See research/16-performance.md for measurements.

        frozen_waves: evaluate the sea once per step instead of at every RK4 stage. About 5x
            faster on CPU; at hs 1 m the trajectories drift by up to a few cm over 4 s.
        compile: run the physics of a step through torch.compile. About 4x faster on MPS
            and on CPU below about 16 envs, slower on CPU at 64 envs and up (inductor's ARM
            cos and sin are scalar). The first step takes about 30 s to compile.
        """
        self.vessel = vessel.to(device)
        self.num_envs = num_envs
        self.sea_state = sea_state or SeaState()
        self.dt = dt
        self.substeps = substeps
        self.episode_length_s = episode_length_s
        self.max_tilt = max_tilt
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.device = torch.device(device)
        self.frozen_waves = frozen_waves
        self._advance = torch.compile(advance, dynamic=False) if compile else advance
        self.generator = torch.Generator().manual_seed(0)

        zeros = torch.zeros(num_envs, 6, device=self.device)
        self.eta, self.nu = zeros.clone(), zeros.clone()
        self.t = torch.zeros(num_envs, device=self.device)
        self.action = torch.zeros(num_envs, vessel.num_actions, device=self.device)
        self.ventilated = torch.zeros(
            num_envs, len(vessel.foils), dtype=torch.bool, device=self.device
        )
        self.wetting_time = torch.zeros(num_envs, len(vessel.foils), device=self.device)
        self.sea = self._sample_sea(num_envs)

    @property
    def observation_size(self) -> int:
        return 12

    def observe(self) -> Tensor:
        return torch.cat((self.eta, self.nu), dim=-1)

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict]:
        if seed is not None:
            self.generator.manual_seed(seed)
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        return self.observe(), {}

    def step(self, action: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Advance one dt. `action=None` is a zero action, for vessels without actions.

        Envs that terminate or truncate reset inside this call: the returned observation is
        already the new episode's first, and the last one is in `info["final_obs"]`.
        """
        if action is None:
            action = torch.zeros(self.num_envs, self.vessel.num_actions)
        self.action = action.clamp(-1, 1).to(self.device)
        self.t, self.eta, self.nu, self.ventilated = self._advance(
            self.vessel,
            self.sea,
            self.t,
            self.eta,
            self.nu,
            self.action,
            self.ventilated,
            self.wetting_time,
            self.dt,
            self.substeps,
            self.frozen_waves,
        )

        reward = self.reward_fn(self)
        tilt = self.eta[:, 3:5].abs().amax(-1)
        terminated = (tilt > self.max_tilt) | ~torch.isfinite(self.eta).all(-1)
        terminated = terminated | self.termination_fn(self)
        truncated = self.t >= self.episode_length_s
        info = {"final_obs": self.observe()}
        done = (terminated | truncated).nonzero().squeeze(-1)
        if len(done):
            self._reset_envs(done)
        return self.observe(), reward, terminated, truncated, info

    def foil_loads(self) -> list[FoilLoads]:
        """Loads on every foil at the current state, for observations, rewards and drawing."""
        pose = Pose.from_eta(self.eta)
        return all_foil_loads(
            self.vessel, self.sea, self.t, pose, self.nu, self.action, self.ventilated
        )

    def _sample_sea(self, num: int) -> Sea:
        s = self.sea_state
        return make_sea(
            num,
            s.hs,
            s.tp,
            heading_rad=s.heading_rad,
            spreading=s.spreading,
            gamma=s.gamma,
            num_components=s.num_components,
            generator=self.generator,
            device=self.device,
        )

    def _reset_envs(self, env_ids: Tensor) -> None:
        self.eta[env_ids] = torch.tensor(self.vessel.initial_eta, device=self.device)
        self.nu[env_ids] = torch.tensor(self.vessel.initial_nu, device=self.device)
        self.t[env_ids] = 0.0
        self.ventilated[env_ids] = False
        self.wetting_time[env_ids] = 0.0
        self.sea.replace(env_ids, self._sample_sea(len(env_ids)))
