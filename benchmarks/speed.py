"""Time N Moths flown by the wand in head seas and print env-steps per second.

    OMP_NUM_THREADS=2 uv run benchmarks/speed.py --num-envs 1024

Real-time factor = env-steps/s x dt: the seconds of simulated vessel time per wall second.
"""

import argparse
import math
import time

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import moth
from watergym.vessels.moth_vessel import wand_action

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=250, help="timed steps, after 25 warm-up steps")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

head_seas = SeaState(hs=0.3, tp=3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
env = WaterEnv(moth(), args.num_envs, head_seas, device=args.device)
env.reset(seed=0)


def run(steps: int) -> None:
    for _ in range(steps):
        env.step(wand_action(env.sea, env.t, env.eta))


run(25)
start = time.perf_counter()
run(args.steps)
if env.device.type == "cuda":
    torch.cuda.synchronize()
wall = time.perf_counter() - start

env_steps_per_s = args.steps * args.num_envs / wall
print(f"{args.num_envs} Moths, wand policy, device {args.device}, torch threads {torch.get_num_threads()}")
print(f"{env_steps_per_s:.0f} env-steps/s, dt {env.dt} s, real-time factor {env_steps_per_s * env.dt:.0f}")
