"""BlueROV2s holding four depths under waves with a PD depth controller. The shallow ones
get pushed around by orbital velocity; it fades as e^(-k z) with depth.

    uv run --extra viz examples/04_underwater_vehicle.py
    uv run --extra viz examples/04_underwater_vehicle.py --headless --seconds 5 \
        --screenshot docs/images/04_underwater_vehicle.png
"""

import argparse

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import bluerov2
from watergym.viewer import WaterViewer, make_viewer

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--screenshot", help="save the last frame as PNG (gl only)")
parser.add_argument("--seconds", type=float, default=30.0)
args = parser.parse_args()

target_depth = torch.tensor([0.5, 1.0, 2.0, 4.0])
env = WaterEnv(bluerov2(), num_envs=4, sea_state=SeaState(hs=0.6, tp=4.0, spreading=10.0))
env.reset(seed=0)
env.eta[:, 2] = target_depth

viewer = make_viewer(args.viewer, headless=args.headless)
scene = WaterViewer(viewer, num_envs=4, patch_size_m=6.0, patch_resolution=48)
scene.look_at_grid(distance=1.0, pitch_deg=-28.0, yaw_deg=45.0)

while viewer.is_running() and env.t[0] < args.seconds:
    depth_error = target_depth - env.eta[:, 2]
    vertical = (2.0 * depth_error - 1.0 * env.nu[:, 2]).clamp(-1, 1)
    action = torch.cat((torch.zeros(4, 4), vertical[:, None].expand(4, 4)), dim=-1)
    env.step(action)
    scene.draw(env.t, env.sea, env.vessel, env.eta)
    for i, depth in enumerate(target_depth.tolist()):
        viewer.log_scalar(f"depth error at {depth} m [cm]", 100 * depth_error[i])
if args.screenshot:
    scene.save_png(args.screenshot)
viewer.close()
