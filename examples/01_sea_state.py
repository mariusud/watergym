r"""Four JONSWAP seas side by side: same Hs, Tp and spreading, different random phases.

    uv run --extra viz examples/01_sea_state.py
    uv run --extra viz examples/01_sea_state.py --headless --seconds 5 \
        --screenshot docs/images/01_sea_state.png
"""

import argparse

import torch

from watergym.viewer import WaterViewer, make_viewer
from watergym.waves import elevation, make_sea, significant_wave_height

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "null"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--screenshot", help="save the last frame as PNG (gl only)")
parser.add_argument("--seconds", type=float, default=20.0)
args = parser.parse_args()

hs, tp = 1.5, 4.5
sea = make_sea(4, hs, tp, spreading=8.0, num_components=96)
print(f"requested Hs {hs} m, realised per env {significant_wave_height(sea).tolist()}")

viewer = make_viewer(args.viewer, headless=args.headless)
scene = WaterViewer(viewer, num_envs=4, patch_size_m=40.0, patch_resolution=128)
scene.look_at_grid(distance=0.7, pitch_deg=-22.0)

dt = 1 / 30
t = torch.zeros(4)
origin = torch.zeros(4, 1, 3)
while viewer.is_running() and t[0] < args.seconds:
    scene.draw(t, sea)
    viewer.log_scalar("elevation at origin, env 0 [m]", elevation(sea, origin, t)[0, 0])
    t += dt
if args.screenshot:
    scene.save_png(args.screenshot)
viewer.close()
