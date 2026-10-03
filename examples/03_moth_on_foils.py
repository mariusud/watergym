"""A fleet of Moths foiling through head seas, ride height held by the mechanical wand.
Orange arrows are the lift and drag on every foil strip.

    uv run --extra viz examples/03_moth_on_foils.py
    uv run --extra viz examples/03_moth_on_foils.py --headless --seconds 5 \
        --screenshot docs/images/03_moth_on_foils.png
"""

import argparse
import math
import time

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import moth
from watergym.vessels.moth_vessel import HULL_BOTTOM_Z, wand_action
from watergym.viewer import WaterViewer, make_viewer

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--screenshot", help="save the last frame as PNG (gl only)")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--hs", type=float, default=0.3, help="significant wave height [m]")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

head_seas = SeaState(hs=args.hs, tp=3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
env = WaterEnv(moth(), args.num_envs, head_seas, device=args.device)
env.reset(seed=0)

viewer = make_viewer(args.viewer, headless=args.headless)
scene = WaterViewer(viewer, args.num_envs, patch_size_m=8.0, patch_resolution=48)
scene.look_at_grid(distance=0.9, pitch_deg=-12.0, yaw_deg=70.0)

ride_heights = []
steps = 0
start = time.perf_counter()
while viewer.is_running() and env.t[0] < args.seconds:
    action = wand_action(env.sea, env.t, env.eta)
    env.step(action)
    ride_heights.append((-env.eta[:, 2] - HULL_BOTTOM_Z).mean())
    steps += 1
    scene.draw(env.t, env.sea, env.vessel, env.eta, env.foil_loads(), newton_per_m=300.0)
    viewer.log_scalar("hull bottom above mean water, env 0 [m]", -env.eta[0, 2] - HULL_BOTTOM_Z)
    viewer.log_scalar(
        "flap, env 0 [deg]", math.degrees(action[0, 1] * env.vessel.foils[0].max_flap)
    )
    viewer.log_scalar("speed, env 0 [m/s]", env.nu[0, 0])
if args.screenshot:
    scene.save_png(args.screenshot)
viewer.close()

if steps:
    wall = time.perf_counter() - start
    print(
        f"{steps * env.dt:.1f} sim s, mean hull bottom above mean water "
        f"{torch.stack(ride_heights).mean():.2f} m, "
        f"{steps * args.num_envs / wall:.0f} env-steps/s"
    )
