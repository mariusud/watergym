"""Box barges heaving and rolling in beam seas: buoyancy and Froude-Krylov forces from
volume samples against the moving surface, nothing else.

    uv run --extra viz examples/02_floating_box.py
    uv run --extra viz examples/02_floating_box.py --headless --seconds 5 \
        --screenshot docs/images/02_floating_box.png
"""

import argparse
import math

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import box_barge
from watergym.viewer import WaterViewer, make_viewer

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--screenshot", help="save the last frame as PNG (gl only)")
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

beam_seas = SeaState(hs=1.5, tp=5.0, heading=math.pi / 2, spreading=20.0)
env = WaterEnv(box_barge(), args.num_envs, beam_seas, dt=0.05, device=args.device)
env.reset(seed=0)
no_action = torch.zeros(args.num_envs, 0, device=args.device)

viewer = make_viewer(args.viewer, headless=args.headless)
scene = WaterViewer(viewer, args.num_envs, patch_size_m=30.0, patch_resolution=64)
scene.look_at_grid(distance=0.8, pitch_deg=-20.0, yaw_deg=30.0)

while viewer.is_running() and env.t[0] < args.seconds:
    env.step(no_action)
    scene.draw(env.t, env.sea, env.vessel, env.eta)
    viewer.log_scalar("heave env 0 [m]", -env.eta[0, 2])
    viewer.log_scalar("roll env 0 [deg]", math.degrees(env.eta[0, 3]))
if args.screenshot:
    scene.save_png(args.screenshot)
viewer.close()
