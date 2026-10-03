"""Adapter that lets rsl_rl's OnPolicyRunner train a WaterEnv (rsl-rl-lib 5.x VecEnv interface).

runner = OnPolicyRunner(RslRlVecEnv(env), train_cfg, log_dir, device)
"""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from watergym.env import WaterEnv


class RslRlVecEnv(VecEnv):
    """rsl_rl sees one observation group, "policy", and a done flag per env.

    WaterEnv resets finished envs inside step(), which is also what rsl_rl expects, so the
    observation returned for a finished env is already the next episode's first one.
    `dones` is terminated or truncated. `extras["time_outs"]` marks the truncated envs so PPO
    bootstraps their return from the value function instead of treating them as failures.
    """

    def __init__(self, env: WaterEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.vessel.num_actions
        self.device = env.device
        self.max_episode_length = math.ceil(env.episode_length_s / env.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {"num_envs": self.num_envs, "dt": env.dt, "vessel": env.vessel.name}
        env.reset()

    def get_observations(self) -> TensorDict:
        return self._wrap(self.env.observe())

    def step(self, actions: Tensor) -> tuple[TensorDict, Tensor, Tensor, dict]:
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        dones = terminated | truncated
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        extras = {"time_outs": (truncated & ~terminated).float()}
        return self._wrap(obs), reward, dones.long(), extras

    def _wrap(self, obs: Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
