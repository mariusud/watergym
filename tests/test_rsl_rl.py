import pytest
import torch

pytest.importorskip("rsl_rl")

from watergym import SeaState, WaterEnv  # noqa: E402
from watergym.rl.rsl_rl import RslRlVecEnv  # noqa: E402
from watergym.vessels import moth  # noqa: E402


def make(num_envs: int = 3, **kwargs) -> RslRlVecEnv:
    return RslRlVecEnv(WaterEnv(moth(), num_envs, SeaState(hs=0.3, tp=3.0), **kwargs))


def test_interface_shapes_and_dtypes() -> None:
    vec = make()
    assert (vec.num_envs, vec.num_actions) == (3, 2)
    assert vec.episode_length_buf.shape == (3,)
    obs = vec.get_observations()
    assert obs["policy"].shape == (3, 12)
    obs, reward, dones, extras = vec.step(torch.zeros(3, 2))
    assert obs["policy"].shape == (3, 12) and obs["policy"].dtype == torch.float32
    assert reward.shape == dones.shape == extras["time_outs"].shape == (3,)
    assert dones.dtype == torch.long


def test_timeout_is_done_and_marked_as_time_out() -> None:
    vec = make(2, episode_length_s=0.1, dt=0.05)
    assert vec.max_episode_length == 2
    vec.step(torch.zeros(2, 2))
    assert vec.episode_length_buf.tolist() == [1, 1]
    _, _, dones, extras = vec.step(torch.zeros(2, 2))
    assert dones.tolist() == [1, 1]
    assert extras["time_outs"].tolist() == [1.0, 1.0]
    assert vec.episode_length_buf.tolist() == [0, 0]


def test_termination_is_done_but_not_a_time_out() -> None:
    vec = make(2)
    vec.env.eta[0, 3] = 2.0  # env 0 rolls past max_tilt
    _, _, dones, extras = vec.step(torch.zeros(2, 2))
    assert dones.tolist() == [1, 0]
    assert extras["time_outs"].tolist() == [0.0, 0.0]
    assert vec.episode_length_buf.tolist() == [0, 1]
    assert vec.env.eta[0, 3] == 0.0


def test_task_gives_policy_and_privileged_critic_groups() -> None:
    from watergym.tasks import RideControlMoth

    vec = RslRlVecEnv(RideControlMoth(2), seed=0)
    assert vec.num_actions == 1
    assert vec.max_episode_length == 1000
    obs = vec.get_observations()
    assert obs["policy"].shape == (2, RideControlMoth.observation_size)
    assert obs["critic"].shape == (2, RideControlMoth.privileged_observation_size)
    obs, _, dones, extras = vec.step(torch.zeros(2, 1))
    assert torch.isfinite(obs["critic"]).all()
    assert dones.tolist() == [0, 0]
    assert "crash_rate" in extras["log"]


def test_task_crash_is_done_but_not_a_time_out() -> None:
    from watergym.tasks import RideControlMoth

    vec = RslRlVecEnv(RideControlMoth(2), seed=0)
    vec.env.env.eta[0, 4] = 0.5  # env 0 pitches past the task's 15 degrees
    _, _, dones, extras = vec.step(torch.zeros(2, 1))
    assert dones.tolist() == [1, 0]
    assert extras["time_outs"].tolist() == [0.0, 0.0]


def test_load_policy_matches_the_runner_actor(tmp_path) -> None:
    from rsl_rl.runners import OnPolicyRunner

    from watergym.rl.rsl_rl import TRAIN_CFG, load_policy
    from watergym.tasks import RideControlMoth

    vec = RslRlVecEnv(RideControlMoth(3), seed=0)
    runner = OnPolicyRunner(vec, TRAIN_CFG, device="cpu")
    runner.save(str(tmp_path / "model.pt"))
    obs = vec.get_observations()
    expected = runner.get_inference_policy()(obs)
    policy = load_policy(tmp_path / "model.pt", RideControlMoth.observation_size, 1)
    assert torch.allclose(policy(obs["policy"]), expected)
