import math

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import moth
from watergym.vessels.moth_vessel import wand_action


def ride_height(env: WaterEnv) -> torch.Tensor:
    """1 at the starting ride height, 0.37 at 10 cm off. NED z points down."""
    error = env.eta[:, 2] - env.vessel.initial_eta[2]
    return torch.exp(-((error / 0.1) ** 2))


head_seas = SeaState(hs=0.3, tp=3.0, heading_rad=math.pi)  # 0.3 m waves on the bow
env = WaterEnv(moth(), num_envs=64, sea_state=head_seas, reward_fn=ride_height)
obs, info = env.reset(seed=0)

for _ in range(250):  # 5 s at dt = 0.02 s
    action = wand_action(env.sea, env.t, env.eta)  # [64, 2]: sail, flap. Your policy goes here.
    obs, reward, terminated, truncated, info = env.step(action)

print(obs.shape, f"mean reward {reward.mean():.2f}")
