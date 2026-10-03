"""Adapter that lets rsl_rl's OnPolicyRunner train a WaterEnv or a task (rsl-rl-lib 5.x VecEnv).

runner = OnPolicyRunner(RslRlVecEnv(env), TRAIN_CFG, log_dir, device)
"""

import math
from collections.abc import Callable
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_class
from tensordict import TensorDict
from torch import Tensor

from watergym.env import WaterEnv
from watergym.tasks import RideControlMoth

# PPO settings the example trains with. The critic reads the "critic" group: a task's
# privileged observations, or the policy observations for a plain WaterEnv.
TRAIN_CFG = {
    "num_steps_per_env": 24,
    "save_interval": 50,
    "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "algorithm": {"class_name": "PPO", "learning_rate": 1e-3},
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128],
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
    },
    "critic": {"class_name": "MLPModel", "hidden_dims": [128, 128], "obs_normalization": True},
}


class RslRlVecEnv(VecEnv):
    """rsl_rl sees two observation groups, "policy" and "critic", and a done flag per env.

    For a task, "policy" is its sensor observations and "critic" its privileged ones. For a
    plain WaterEnv both are `env.observe()`.

    Finished envs reset inside step(), which is also what rsl_rl expects, so the observation
    returned for a finished env is already the next episode's first one. `dones` is
    terminated or truncated. `extras["time_outs"]` marks envs that were truncated and not
    terminated, so PPO bootstraps their return from the value function instead of treating
    them as failures.
    """

    def __init__(self, env: WaterEnv | RideControlMoth, seed: int | None = None) -> None:
        self.env = env
        self.water_env = env if isinstance(env, WaterEnv) else env.env
        self.num_envs = env.num_envs
        self.num_actions = env.vessel.num_actions if isinstance(env, WaterEnv) else env.action_size
        self.device = env.device
        self.max_episode_length = math.ceil(self.water_env.episode_length_s / self.water_env.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {
            "num_envs": self.num_envs,
            "dt": self.water_env.dt,
            "vessel": self.water_env.vessel.name,
            "task": getattr(env, "task_id", None),
        }
        obs, info = env.reset(seed)
        self._obs = self._wrap(obs, info)

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: Tensor) -> tuple[TensorDict, Tensor, Tensor, dict]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = terminated | truncated
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        extras = {"time_outs": (truncated & ~terminated).float()}
        if "signals" in info:
            signals = info["signals"]
            extras["signals"] = signals
            extras["log"] = {
                "crash_rate": signals["crashed"].float().mean(),
                "abs_height_error_m": signals["height_error_m"].abs().mean(),
            }
        self._obs = self._wrap(obs, info)
        return self._obs, reward, dones.long(), extras

    def _wrap(self, obs: Tensor, info: dict) -> TensorDict:
        critic = info.get("privileged_obs", obs)
        return TensorDict(
            {"policy": obs, "critic": critic}, batch_size=[self.num_envs], device=self.device
        )


def load_policy(
    checkpoint: str | Path, observation_size: int, num_actions: int, cfg: dict = TRAIN_CFG
) -> Callable[[Tensor], Tensor]:
    """The deterministic actor of an rsl_rl checkpoint as a callable obs [envs, n] -> actions."""
    actor_class, actor_cfg = resolve_class(cfg["actor"])
    example = TensorDict({"policy": torch.zeros(1, observation_size)}, batch_size=[1])
    actor = actor_class(example, {"actor": ["policy"]}, "actor", num_actions, **actor_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    actor.load_state_dict(state)
    actor.eval()

    def policy(obs: Tensor) -> Tensor:
        with torch.no_grad():
            return actor(TensorDict({"policy": obs.cpu()}, batch_size=[obs.shape[0]]))

    return policy
