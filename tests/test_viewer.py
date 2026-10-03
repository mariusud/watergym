import math

import pytest
import torch

pytest.importorskip("newton")

from watergym.env import SeaState, WaterEnv  # noqa: E402
from watergym.rigid_body import Pose  # noqa: E402
from watergym.vessels import moth  # noqa: E402
from watergym.viewer import WaterViewer, make_viewer, quaternion_xyzw  # noqa: E402


def test_quaternion_of_a_quarter_turn_about_z() -> None:
    rotation = Pose.from_eta(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]])).rotation
    s = math.sqrt(0.5)
    assert quaternion_xyzw(rotation)[0].tolist() == pytest.approx([0.0, 0.0, s, s], abs=1e-6)


def test_a_frame_draws_without_a_window() -> None:
    env = WaterEnv(moth(), 2, SeaState(hs=0.3, tp=3.0))
    env.reset(seed=0)
    scene = WaterViewer(make_viewer("null"), 2, patch_resolution=8)
    scene.draw(env.t, env.sea, env.vessel, env.eta, env.foil_loads())
