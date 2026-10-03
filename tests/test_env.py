import math

import torch

from watergym.env import SeaState, WaterEnv
from watergym.vessels import bluerov2, box_barge, moth, otter
from watergym.vessels.moth import wand_action


def test_every_vessel_steps_with_gymnasium_shapes() -> None:
    for vessel in (box_barge(), moth(), otter(), bluerov2()):
        env = WaterEnv(vessel, 3, SeaState(hs=0.5, tp=5.0))
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(torch.zeros(3, vessel.num_actions))
        assert obs.shape == (3, env.observation_size)
        assert reward.shape == terminated.shape == truncated.shape == (3,)
        assert torch.isfinite(obs).all()


def test_reset_with_seed_repeats_the_sea() -> None:
    env = WaterEnv(box_barge(), 2, SeaState(hs=1.0, tp=6.0))
    env.reset(seed=7)
    first = env.sea.phase.clone()
    env.reset(seed=7)
    assert torch.equal(first, env.sea.phase)
    assert not torch.equal(first[0], first[1])


def test_truncated_envs_restart() -> None:
    env = WaterEnv(box_barge(), 2, episode_length_s=0.1, dt=0.05)
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step(torch.zeros(2, 0))
    assert truncated.all()
    assert (env.t == 0).all()


def test_wand_keeps_the_moth_flying_in_calm_water() -> None:
    env = WaterEnv(moth(), 1, SeaState(hs=0.0))
    env.reset()
    for _ in range(150):
        env.step(wand_action(env.sea, env.t, env.eta))
    hull_bottom_height = -(env.eta[0, 2].item() + 0.25)
    assert 0.2 < hull_bottom_height < 0.9
    assert abs(env.eta[0, 4].item()) < math.radians(3)
