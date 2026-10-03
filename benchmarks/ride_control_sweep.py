"""Evaluate policies on RideControl-Moth-v0 over a grid of sea states.

    OMP_NUM_THREADS=2 uv run --with matplotlib python benchmarks/ride_control_sweep.py --quick

A policy is any callable from observations [envs, 6] to a flap command [envs, 1]: the
built-in "wand" and "zero", or "package.module:attribute" for your own.
"""

import argparse
import csv
import importlib
import itertools
import math
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from watergym.tasks import make
from watergym.tasks.metrics import summarize
from watergym.tasks.ride_control import (
    DT,
    EVAL_SEEDS,
    TASK_ID,
    sea_state,
    wand_policy,
    zero_policy,
)

Policy = Callable[[Tensor], Tensor]

FULL_GRID = {
    "hs": (0.0, 0.15, 0.3, 0.45, 0.6),
    "tp": (2.5, 4.0),
    "heading_rad": (math.pi, 0.0),
}
QUICK_GRID = {"hs": (0.0, 0.2, 0.4, 0.6), "tp": (3.0,), "heading_rad": (math.pi,)}
BASELINES = {"wand": wand_policy, "zero": zero_policy}


def load_policy(name: str) -> Policy:
    if name in BASELINES:
        return BASELINES[name]
    module, attribute = name.split(":")
    return getattr(importlib.import_module(module), attribute)


@torch.no_grad()
def evaluate(
    policy: Policy,
    hs: float,
    tp: float,
    heading_rad: float,
    seeds: tuple[int, ...],
    num_envs: int,
    seconds: float,
) -> dict[str, float]:
    """Metrics for one sea state, pooled over every env and seed."""
    runs = []
    for seed in seeds:
        task = make(TASK_ID, num_envs, sea_state(hs, tp, heading_rad))
        obs, _ = task.reset(seed=seed)
        steps = []
        for _ in range(round(seconds / DT)):
            obs, _, _, _, info = task.step(policy(obs))
            steps.append(info["signals"])
        runs.append({k: torch.stack([s[k] for s in steps]) for k in steps[0]})
    pooled = {k: torch.cat([run[k] for run in runs], dim=1) for k in runs[0]}
    return summarize(pooled, DT)


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [k for k in rows[0] if k not in ("policy", "hs", "tp", "heading_rad")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    curves = sorted({(r["policy"], r["tp"], r["heading_rad"]) for r in rows})
    for ax, metric in zip(axes.flat, metrics, strict=True):
        for key in curves:
            policy, tp, heading = key
            curve = [r for r in rows if (r["policy"], r["tp"], r["heading_rad"]) == key]
            label = f"{policy}, Tp {tp:g} s, heading {math.degrees(heading):.0f} deg"
            ax.plot([r["hs"] for r in curve], [r[metric] for r in curve], "o-", label=label)
        ax.set_xlabel("Hs [m]")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{TASK_ID} across sea states")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", nargs="+", default=["wand", "zero"])
    parser.add_argument("--quick", action="store_true", help="4 sea states, 4 envs, 10 s, 1 seed")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--csv", type=Path, default=Path("benchmarks/results/ride_control.csv"))
    parser.add_argument("--plot", type=Path, default=Path("docs/images/benchmark_ride_control.png"))
    args = parser.parse_args()

    grid, seeds = FULL_GRID, EVAL_SEEDS
    if args.quick:
        grid, seeds, args.num_envs, args.seconds = QUICK_GRID, EVAL_SEEDS[:1], 4, 10.0

    rows = []
    for name in args.policy:
        policy = load_policy(name)
        for hs, tp, heading in itertools.product(grid["hs"], grid["tp"], grid["heading_rad"]):
            metrics = evaluate(policy, hs, tp, heading, seeds, args.num_envs, args.seconds)
            rows.append({"policy": name, "hs": hs, "tp": tp, "heading_rad": heading, **metrics})
            print(", ".join(f"{k} {v:.3g}" for k, v in metrics.items()), f"<- {name} Hs {hs}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.plot)


if __name__ == "__main__":
    main()
