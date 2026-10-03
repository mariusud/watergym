import math

import torch

from watergym.env import SeaState, WaterEnv
from watergym.vessels import bluerov2, box_barge, moth, moth_vessel, otter
from watergym.vessels.moth_vessel import wand_action


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
        _, _, _, truncated, _ = env.step(None)
    assert truncated.all()
    assert (env.t == 0).all()


def test_step_none_is_a_zero_action() -> None:
    envs = [WaterEnv(moth(), 2, SeaState(hs=0.3, tp=3.0)) for _ in range(2)]
    for env in envs:
        env.reset(seed=0)
    envs[0].step(None)
    envs[1].step(torch.zeros(2, envs[1].vessel.num_actions))
    assert torch.equal(envs[0].eta, envs[1].eta)


def test_wand_keeps_the_moth_flying_in_calm_water() -> None:
    env = WaterEnv(moth(), 1, SeaState(hs=0.0))
    env.reset()
    for _ in range(150):
        env.step(wand_action(env.sea, env.t, env.eta))
    hull_bottom_height = -(env.eta[0, 2].item() + 0.25)
    assert 0.2 < hull_bottom_height < 0.9
    assert abs(env.eta[0, 4].item()) < math.radians(3)


def test_vessel_module_constants_are_reachable_beside_the_factory() -> None:
    assert moth_vessel.moth is moth
    assert moth_vessel.WAND_GEARING > 0


def test_ventilated_foils_rewet_after_the_washout_time() -> None:
    env = WaterEnv(moth(), num_envs=1, sea_state=SeaState(hs=0.0, tp=5.0))
    env.reset(seed=0)
    for _ in range(100):
        env.step(None)
    env.ventilated[:] = True
    washout_steps = round(env.vessel.foils[0].washout_time_s / env.dt)
    for _ in range(washout_steps - 1):
        env.step(None)
    assert env.ventilated.all()
    env.step(None)
    assert not env.ventilated.any()


def test_termination_fn_ends_and_resets_the_chosen_envs() -> None:
    env = WaterEnv(box_barge(), 3, termination_fn=lambda env: torch.tensor([True, False, False]))
    env.reset(seed=0)
    env.step(torch.zeros(3, 0))
    _, _, terminated, _, _ = env.step(torch.zeros(3, 0))
    assert terminated.tolist() == [True, False, False]
    assert env.t[0] == 0.0
    assert env.t[1] > 0.0


def test_frozen_waves_stay_close_to_the_exact_sea() -> None:
    envs = [
        WaterEnv(box_barge(), 2, SeaState(hs=1.0, tp=6.0), frozen_waves=frozen)
        for frozen in (False, True)
    ]
    for env in envs:
        env.reset(seed=0)
        for _ in range(50):
            env.step(None)
    assert torch.allclose(envs[0].eta, envs[1].eta, atol=1e-3)
    assert not torch.equal(envs[0].eta, envs[1].eta)


def test_episodes_end_on_a_step_count_not_summed_time() -> None:
    # float32 time summed in 0.01 s substeps reaches 1.0 only after 51 steps
    env = WaterEnv(box_barge(), 1, episode_length_s=1.0, dt=0.02)
    env.reset()
    for _ in range(49):
        _, _, _, truncated, _ = env.step(None)
        assert not truncated.any()
    _, _, _, truncated, _ = env.step(None)
    assert truncated.all()
    assert (env.steps == 0).all()


def test_diverged_env_terminates_with_a_finite_reward() -> None:
    env = WaterEnv(box_barge(), 2, reward_fn=lambda env: env.nu[:, 0])
    env.reset(seed=0)
    env.nu[0, 0] = float("nan")
    _, reward, terminated, _, _ = env.step(None)
    assert torch.isfinite(reward).all()
    assert terminated.tolist() == [True, False]
