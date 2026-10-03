"""Train the Moth to hold its ride height with rsl_rl's PPO, or watch a saved checkpoint.

uv run --extra rl examples/train_rsl_rl.py --num-envs 1024 --iterations 300
uv run --extra rl examples/train_rsl_rl.py --task RideControl-Moth-v0 --hs 0.3
uv run --extra rl --extra viz examples/train_rsl_rl.py --play logs/rsl_rl/moth/<run>/model_299.pt
uv run --extra rl --extra viz examples/train_rsl_rl.py --play <ckpt> --viewer null

Without --task the reward is the toy one below and the policy sees the full state. With
--task RideControl-Moth-v0 the policy sees the task's sensor observations, the critic its
privileged ones, and the task's own reward, terminations and 20 s timeout apply.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from watergym import SeaState, WaterEnv
from watergym.rl.rsl_rl import TRAIN_CFG, RslRlVecEnv
from watergym.tasks import TASKS, make
from watergym.tasks.ride_control import EVAL_SEEDS, sea_state
from watergym.vessels import moth

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=list(TASKS), help="train on a benchmark task")
parser.add_argument("--hs", type=float, default=0.3, help="significant wave height [m]")
parser.add_argument("--tp", type=float, default=3.0, help="peak period [s]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--device", default="cpu")
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--play", metavar="CHECKPOINT", help="run this checkpoint instead of training")
parser.add_argument("--seconds", type=float, default=20.0, help="--play length in sim seconds")
args = parser.parse_args()
if args.seed in EVAL_SEEDS:
    parser.error(f"seed {args.seed} is held out for evaluation")
torch.manual_seed(args.seed)


def ride_height_reward(env: WaterEnv) -> torch.Tensor:
    """1 at the starting height, falling to 0.37 at 0.1 m off. Heights are -z in NED."""
    error = -env.eta[:, 2] + env.vessel.initial_eta[2]
    return torch.exp(-((error / 0.1) ** 2))


num_envs = 4 if args.play else args.num_envs
if args.task:
    env = make(args.task, num_envs, sea_state(args.hs, args.tp), device=args.device)
    water_env = env.env
else:
    head_seas = SeaState(
        hs=args.hs, tp=args.tp, heading_rad=math.pi, spreading=10.0, num_components=48
    )
    env = WaterEnv(moth(), num_envs, head_seas, reward_fn=ride_height_reward, device=args.device)
    water_env = env
vec_env = RslRlVecEnv(env, seed=args.seed)

if args.play:
    from watergym.viewer import WaterViewer, make_viewer

    runner = OnPolicyRunner(vec_env, TRAIN_CFG, device=args.device)
    runner.load(args.play, map_location=args.device)
    policy = runner.get_inference_policy()
    viewer = make_viewer(args.viewer)
    scene = WaterViewer(viewer, num_envs, patch_size_m=8.0, patch_resolution=48)
    scene.look_at_grid(distance=0.9, pitch_deg=-12.0, yaw_deg=70.0)
    obs = vec_env.get_observations()
    episode_reward = torch.zeros(num_envs)
    finished_rewards = []
    squared_errors = []
    crashes = 0
    for _ in range(round(args.seconds / water_env.dt)):
        with torch.inference_mode():
            obs, reward, dones, extras = vec_env.step(policy(obs))
        episode_reward += reward
        is_done = dones.bool()
        crashes += int((is_done & (extras["time_outs"] == 0)).sum())
        finished_rewards.extend(episode_reward[is_done].tolist())
        episode_reward[is_done] = 0.0
        if "signals" in extras:  # the task's target ride height
            error = extras["signals"]["height_error_m"]
        else:  # the starting height
            error = -water_env.eta[:, 2] + water_env.vessel.initial_eta[2]
        squared_errors.append((error**2)[~is_done])
        scene.draw(
            water_env.t,
            water_env.sea,
            water_env.vessel,
            water_env.eta,
            water_env.foil_loads(),
            newton_per_m=300.0,
        )
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
    run_dir = args.task or "moth"
    log_dir = Path("logs/rsl_rl") / run_dir / time.strftime("%Y%m%d-%H%M%S")
    runner = OnPolicyRunner(vec_env, TRAIN_CFG, str(log_dir), args.device)
    runner.learn(args.iterations)
    print(f"checkpoints in {log_dir}")
