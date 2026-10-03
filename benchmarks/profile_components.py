"""Speed and response statistics against SeaState.num_components.

    OMP_NUM_THREADS=2 uv run benchmarks/profile_components.py --vessel moth

Different component counts draw different seas, so trajectories cannot be compared one to
one. This compares what an RL agent sees on average: the standard deviation of heave, pitch
and the wave elevation at the CG over many envs, after a 2 s warm-up.
"""

import argparse
import math
import time

import torch

from watergym import SeaState, WaterEnv, vessels
from watergym.vessels.moth_vessel import wand_action
from watergym.waves import elevation

parser = argparse.ArgumentParser()
parser.add_argument("--vessel", default="moth")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--components", type=int, nargs="+", default=[8, 16, 32, 64, 128])
args = parser.parse_args()

hs, tp = (0.3, 3.0) if args.vessel == "moth" else (1.0, 6.0)
print(f"{args.vessel}, hs {hs} tp {tp}, {args.num_envs} envs, {args.steps} steps")
print(f"{'comp':>5} {'env-steps/s':>12} {'std z':>8} {'std pitch':>10} {'std elev':>9} {'resets':>7}")
for num in args.components:
    sea = SeaState(hs, tp, math.pi, 10.0, num_components=num)
    env = WaterEnv(getattr(vessels, args.vessel)(), args.num_envs, sea)
    env.reset(seed=0)
    zero = torch.zeros(args.num_envs, env.vessel.num_actions)
    records, resets = [], 0
    origin = torch.zeros(1, 3)
    start = time.perf_counter()
    for i in range(args.steps):
        action = wand_action(env.sea, env.t, env.eta) if args.vessel == "moth" else zero
        _, _, terminated, truncated, _ = env.step(action)
        resets += int((terminated | truncated).sum())
        if i >= 100:
            wave = elevation(env.sea, env.eta[:, None, :3] * 0 + origin, env.t)[:, 0]
            records.append(torch.stack((env.eta[:, 2], env.eta[:, 4], wave), -1))
    rate = args.steps * args.num_envs / (time.perf_counter() - start)
    std = torch.stack(records).std(0).mean(0)
    print(f"{num:>5} {rate:>12.0f} {std[0]:>8.4f} {math.degrees(std[1]):>9.3f}° {std[2]:>9.4f} "
          f"{resets:>7}")
