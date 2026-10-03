"""Save or check reference trajectories for all four vessels at hs = 1.0.

    uv run benchmarks/profile_trajectory.py save ref.pt
    uv run benchmarks/profile_trajectory.py check ref.pt [--frozen-waves] [--num-components 32]

Prints the largest |difference| in eta and nu over 200 steps, per vessel, absolute and\nrelative to |x| + 1 (float32 resolves positions of 30 m only to about 4e-6).
"""

import argparse
import math

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import bluerov2, box_barge, moth, otter
from watergym.vessels.moth_vessel import wand_action

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("save", "check"))
parser.add_argument("path")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--device", default="cpu")
parser.add_argument("--env-kwargs", default="{}", help="extra WaterEnv kwargs as a Python dict")
parser.add_argument("--perturb", type=float, default=0.0, help="add this to nu after reset")
args = parser.parse_args()


def trajectory(factory) -> tuple[torch.Tensor, torch.Tensor]:
    vessel = factory()
    sea = SeaState(hs=1.0, tp=6.0, heading_rad=math.pi, spreading=10.0)
    env = WaterEnv(vessel, args.num_envs, sea, device=args.device, **eval(args.env_kwargs))
    env.reset(seed=0)
    env.nu += args.perturb
    commands = torch.linspace(-0.5, 0.5, args.num_envs, device=env.device)[:, None]
    states, vents = [], []
    for _ in range(args.steps):
        if vessel.name == "moth":
            action = wand_action(env.sea, env.t, env.eta)
        else:
            action = commands.expand(args.num_envs, vessel.num_actions)
        env.step(action)
        states.append(torch.cat((env.eta, env.nu), -1).double().cpu())
        vents.append(env.ventilated.cpu())
    return torch.stack(states), torch.stack(vents)


results = {f.__name__: trajectory(f) for f in (box_barge, moth, otter, bluerov2)}
if args.mode == "save":
    torch.save(results, args.path)
else:
    reference = torch.load(args.path)
    for name, (states, vents) in results.items():
        ref_states = reference[name][0]
        diff = (states - ref_states).abs()
        worst = diff.flatten(0, 1).amax(0)
        relative = (diff / (ref_states.abs() + 1)).amax().item()
        vent_flips = (vents != reference[name][1]).sum().item()
        per_dof = " ".join(f"{d:.1e}" for d in worst.tolist())
        print(f"{name:10s} max |d| {diff.amax():.2e}  max |d|/(|x|+1) {relative:.2e}  "
              f"vent flips {vent_flips}  per state [{per_dof}]")
