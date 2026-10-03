"""Top operators by self CPU time for a few env steps.

    OMP_NUM_THREADS=2 uv run benchmarks/profile_ops.py --vessel moth --num-envs 1024
"""

import argparse
import math

import torch
from torch.profiler import ProfilerActivity, profile

from watergym import SeaState, WaterEnv, vessels
from watergym.vessels.moth_vessel import wand_action

parser = argparse.ArgumentParser()
parser.add_argument("--vessel", default="moth")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--components", type=int, default=48)
parser.add_argument("--sort", default="self_cpu_time_total")
args = parser.parse_args()

sea = SeaState(0.3, 3.0, math.pi, 10.0, num_components=args.components)
env = WaterEnv(getattr(vessels, args.vessel)(), args.num_envs, sea)
env.reset(seed=0)
zero = torch.zeros(args.num_envs, env.vessel.num_actions)


def step() -> None:
    env.step(wand_action(env.sea, env.t, env.eta) if args.vessel == "moth" else zero)


for _ in range(3):
    step()
with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    for _ in range(args.steps):
        step()
print(prof.key_averages().table(sort_by=args.sort, row_limit=25))
