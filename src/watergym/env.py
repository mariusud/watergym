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
from watergym.rigid_body import Pose, rk4_step
from watergym.vessel import Vessel, all_foil_loads, vessel_derivative
from watergym.waves import Sea, make_sea

RewardFn = Callable[["WaterEnv"], Tensor]


@dataclass
class SeaState:
    hs: float = 0.0
    tp: float = 6.0
    heading: float = 0.0
    spreading: float | None = None
    gamma: float = 3.3
    num_components: int = 64


def no_reward(env: "WaterEnv") -> Tensor:
    return torch.zeros(env.num_envs, device=env.device)


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
        device: str | torch.device = "cpu",
    ) -> None:
        self.vessel = vessel.to(device)
        self.num_envs = num_envs
        self.sea_state = sea_state or SeaState()
        self.dt = dt
        self.substeps = substeps
        self.episode_length_s = episode_length_s
        self.max_tilt = max_tilt
        self.reward_fn = reward_fn
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(0)

        zeros = torch.zeros(num_envs, 6, device=self.device)
        self.eta, self.nu = zeros.clone(), zeros.clone()
        self.t = torch.zeros(num_envs, device=self.device)
        self.action = torch.zeros(num_envs, vessel.num_actions, device=self.device)
        self.ventilated = torch.zeros(
            num_envs, len(vessel.foils), dtype=torch.bool, device=self.device
        )
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

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        self.action = action.clamp(-1, 1).to(self.device)
        derivative = vessel_derivative(self.vessel, self.sea, self.action, self.ventilated)
        substep_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.eta, self.nu = rk4_step(derivative, self.t, self.eta, self.nu, substep_dt)
            self.t = self.t + substep_dt
        self._update_ventilation()

        reward = self.reward_fn(self)
        tilt = self.eta[:, 3:5].abs().amax(-1)
        terminated = (tilt > self.max_tilt) | ~torch.isfinite(self.eta).all(-1)
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

    def _update_ventilation(self) -> None:
        loads = self.foil_loads()
        for i, foil in enumerate(self.vessel.foils):
            self.ventilated[:, i] = update_ventilation(foil, self.ventilated[:, i], loads[i])

    def _sample_sea(self, num: int) -> Sea:
        s = self.sea_state
        return make_sea(
            num,
            s.hs,
            s.tp,
            heading=s.heading,
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
        self.sea.replace(env_ids, self._sample_sea(len(env_ids)))
