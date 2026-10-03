"""Env-steps/s and peak memory per env for one batch size, in a fresh process.

    OMP_NUM_THREADS=2 uv run benchmarks/profile_scaling.py --num-envs 64 1024 4096

Each size runs in its own subprocess so peak RSS belongs to that size alone. Memory per env
is (peak RSS - RSS before the env was built) / num_envs on CPU, and the MPS driver's
allocation (which keeps freed blocks cached, so it tracks the peak) on MPS.
"""

import argparse
import ast
import json
import math
import resource
import subprocess
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 1024, 4096])
parser.add_argument("--device", default="cpu")
parser.add_argument("--vessel", default="moth")
parser.add_argument("--seconds", type=float, default=3.0, help="timed wall time per size")
parser.add_argument("--max-steps", type=int, default=200)
parser.add_argument("--env-kwargs", default="{}", help="extra WaterEnv kwargs as a Python dict")
parser.add_argument("--components", type=int, default=48)
parser.add_argument("--child", action="store_true")
args = parser.parse_args()


def sync() -> None:
    if args.device == "mps":
        torch.mps.synchronize()


def measure(num_envs: int) -> dict:
    from watergym import SeaState, WaterEnv, vessels
    from watergym.vessels.moth_vessel import wand_action

    sea = SeaState(
        1.0 if args.vessel != "moth" else 0.3, 3.0, math.pi, 10.0, num_components=args.components
    )
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    env = WaterEnv(
        getattr(vessels, args.vessel)(),
        num_envs,
        sea,
        device=args.device,
        **ast.literal_eval(args.env_kwargs),
    )
    env.reset(seed=0)
    zero = torch.zeros(num_envs, env.vessel.num_actions, device=env.device)

    def step() -> None:
        action = wand_action(env.sea, env.t, env.eta) if args.vessel == "moth" else zero
        env.step(action)

    for _ in range(5):
        step()
    sync()
    steps, start = 0, time.perf_counter()
    while steps < args.max_steps and (steps < 3 or time.perf_counter() - start < args.seconds):
        step()
        steps += 1
    sync()
    wall = time.perf_counter() - start
    if args.device == "mps":
        memory = torch.mps.driver_allocated_memory()
    else:
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before
    return {
        "num_envs": num_envs,
        "steps_per_s": steps / wall,
        "env_steps_per_s": steps * num_envs / wall,
        "bytes_per_env": memory / num_envs,
    }


if args.child:
    print(json.dumps(measure(args.num_envs[0])))
else:
    threads = torch.get_num_threads()
    print(f"{args.vessel} on {args.device}, {args.components} components, threads {threads}")
    print(f"{'envs':>6} {'steps/s':>9} {'env-steps/s':>12} {'kB/env':>8}")
    for n in args.num_envs:
        cmd = [sys.executable, *sys.argv, "--child", "--num-envs", str(n)]
        result = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        print(
            f"{n:>6} {result['steps_per_s']:>9.1f} {result['env_steps_per_s']:>12.0f} "
            f"{result['bytes_per_env'] / 1e3:>8.1f}",
            flush=True,
        )
        if args.device == "mps":
            time.sleep(3)
