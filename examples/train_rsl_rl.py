"""Train the Moth to hold its ride height with rsl_rl's PPO, or watch a saved checkpoint.

uv run --extra rl examples/train_rsl_rl.py --num-envs 1024 --iterations 300
uv run --extra rl --extra viz examples/train_rsl_rl.py --play logs/rsl_rl/moth/<run>/model_299.pt
uv run --extra rl --extra viz examples/train_rsl_rl.py --play <checkpoint> --viewer null --seconds 20
"""

import argparse
import math
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from watergym import SeaState, WaterEnv
from watergym.rl.rsl_rl import RslRlVecEnv
from watergym.vessels import moth

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--device", default="cpu")
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--play", metavar="CHECKPOINT", help="run this checkpoint instead of training")
parser.add_argument("--seconds", type=float, default=20.0, help="--play length in sim seconds")
args = parser.parse_args()


def ride_height_reward(env: WaterEnv) -> torch.Tensor:
    """1 at the starting height, falling to 0.37 at 0.1 m off. Heights are -z in NED."""
    error = -env.eta[:, 2] + env.vessel.initial_eta[2]
    return torch.exp(-((error / 0.1) ** 2))


head_seas = SeaState(hs=0.3, tp=3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
num_envs = 4 if args.play else args.num_envs
env = WaterEnv(moth(), num_envs, head_seas, reward_fn=ride_height_reward, device=args.device)

train_cfg = {
    "num_steps_per_env": 24,
    "save_interval": 50,
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "algorithm": {"class_name": "PPO", "learning_rate": 1e-3},
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128],
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
    },
    "critic": {"class_name": "MLPModel", "hidden_dims": [128, 128], "obs_normalization": True},
}

if args.play:
    from watergym.viewer import WaterViewer, make_viewer

    runner = OnPolicyRunner(RslRlVecEnv(env), train_cfg, device=args.device)
    runner.load(args.play, map_location=args.device)
    policy = runner.get_inference_policy()
    viewer = make_viewer(args.viewer)
    scene = WaterViewer(viewer, num_envs, patch_size_m=8.0, patch_resolution=48)
    scene.look_at_grid(distance=0.9, pitch_deg=-12.0, yaw_deg=70.0)
    obs = runner.env.get_observations()
    episode_reward = torch.zeros(num_envs)
    finished_rewards = []
    squared_errors = []
    crashes = 0
    for _ in range(round(args.seconds / env.dt)):
        with torch.inference_mode():
            obs, reward, dones, extras = runner.env.step(policy(obs))
        episode_reward += reward
        is_done = dones.bool()
        crashes += int((is_done & (extras["time_outs"] == 0)).sum())
        finished_rewards.extend(episode_reward[is_done].tolist())
        episode_reward[is_done] = 0.0
        squared_errors.append(((-env.eta[:, 2] + env.vessel.initial_eta[2]) ** 2)[~is_done])
        scene.draw(env.t, env.sea, env.vessel, env.eta, env.foil_loads(), newton_per_m=300.0)
        if not viewer.is_running():
            break
    viewer.close()
    episode_rewards = finished_rewards + episode_reward.tolist()
    ride_height_rms = torch.cat(squared_errors).mean().sqrt()
    print(
        f"{args.seconds:.0f} s x {num_envs} envs: "
        f"mean episode reward {sum(episode_rewards) / len(episode_rewards):.1f}, "
        f"ride-height RMS {ride_height_rms:.3f} m, crashes {crashes}"
    )
else:
    log_dir = Path("logs/rsl_rl/moth") / time.strftime("%Y%m%d-%H%M%S")
    runner = OnPolicyRunner(RslRlVecEnv(env), train_cfg, str(log_dir), args.device)
    runner.learn(args.iterations)
    print(f"checkpoints in {log_dir}")
