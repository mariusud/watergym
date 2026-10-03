import math

import pytest
import torch

from watergym.tasks import TASKS, make, ride_control
from watergym.tasks.metrics import analog_gain, per_minute, rms, wk_analog_sections, wk_filter
from watergym.tasks.ride_control import TASK_ID, sea_state, wand_policy, zero_policy
from watergym.vessels.moth_vessel import wand_action

# ISO 2631-1:1997 Table 3, Wk weighting factors (x1000) at one-third-octave frequencies.
ISO_WK_TABLE = {
    0.1: 31.2,
    0.2: 121,
    0.5: 418,
    1.0: 482,
    2.0: 531,
    4.0: 967,
    5.0: 1039,
    8.0: 1036,
    16.0: 768,
    31.5: 405,
}


def run(policy, steps: int, hs: float = 0.0, num_envs: int = 2, seed: int = 0) -> list[dict]:
    task = make(TASK_ID, num_envs, sea_state(hs, 3.0))
    obs, _ = task.reset(seed=seed)
    signals = []
    for _ in range(steps):
        obs, _, _, _, info = task.step(policy(obs))
        signals.append(info["signals"])
    return signals


@pytest.mark.parametrize("frequency", ISO_WK_TABLE)
def test_wk_analog_weighting_matches_the_iso_table(frequency: float) -> None:
    expected = ISO_WK_TABLE[frequency] / 1000
    assert analog_gain(wk_analog_sections(), frequency) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("frequency", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_wk_filter_weights_a_sine_like_the_iso_table(frequency: float) -> None:
    t = torch.arange(0, 60, 0.02)[:, None]
    weighted = wk_filter(torch.sin(2 * math.pi * frequency * t), sample_rate=50.0)
    gain = rms(weighted[1000:]) * math.sqrt(2)
    assert gain == pytest.approx(ISO_WK_TABLE[frequency] / 1000, rel=0.03)


def test_per_minute_counts_events_over_sailed_time() -> None:
    events = torch.zeros(3000, 2, dtype=torch.bool)
    events[10, 0] = events[20, 1] = True
    assert per_minute(events, dt=0.02) == pytest.approx(1.0)


def test_task_shapes() -> None:
    task = make(TASK_ID, 3, sea_state(0.3, 3.0))
    obs, info = task.reset(seed=0)
    assert obs.shape == (3, task.observation_size)
    assert info["privileged_obs"].shape == (3, task.privileged_observation_size)
    obs, reward, terminated, truncated, info = task.step(torch.zeros(3, 1))
    assert reward.shape == terminated.shape == truncated.shape == (3,)
    assert info["final_obs"].shape == obs.shape
    assert torch.isfinite(info["privileged_obs"]).all()


def test_wand_policy_on_the_bow_sensor_is_the_mechanical_wand() -> None:
    task = make(TASK_ID, 2, sea_state(0.4, 3.0))
    obs, _ = task.reset(seed=1)
    for _ in range(20):
        obs, *_ = task.step(wand_policy(obs))
    env = task.env
    expected = wand_action(env.sea, env.t, env.eta)[:, 1:]
    assert torch.allclose(wand_policy(obs), expected, atol=1e-5)


def test_wand_flies_and_zero_flap_touches_down() -> None:
    wand = run(wand_policy, 300)
    zero = run(zero_policy, 400)
    assert not any(s["crashed"].any() for s in wand)
    assert any(s["touchdown"].any() for s in zero)


def test_evaluation_is_deterministic_for_a_seed() -> None:
    first, second = (run(wand_policy, 50, hs=0.4, seed=7) for _ in range(2))
    assert torch.equal(first[-1]["height_error_m"], second[-1]["height_error_m"])


def test_v0_definition_is_frozen() -> None:
    """Changing any of these changes RideControl-Moth-v0. Make RideControl-Moth-v1 instead."""
    assert set(TASKS) == {"RideControl-Moth-v0"}
    frozen = (
        ride_control.DT,
        ride_control.EPISODE_LENGTH_S,
        ride_control.SAIL_COMMAND,
        ride_control.TARGET_RIDE_HEIGHT_M,
        ride_control.MAX_PITCH,
        ride_control.SPREADING,
        ride_control.NUM_WAVE_COMPONENTS,
        ride_control.EVAL_SEEDS,
        (ride_control.W_HEIGHT, ride_control.HEIGHT_SCALE_M),
        (ride_control.W_PITCH, ride_control.PITCH_SCALE),
        ride_control.W_EFFORT,
        ride_control.W_SMOOTHNESS,
        ride_control.W_COMFORT,
        ride_control.W_CRASH,
        ride_control.POLICY_OBSERVATIONS,
        len(ride_control.PRIVILEGED_OBSERVATIONS),
    )
    assert frozen == (
        0.02,
        20.0,
        0.65,
        0.6,
        math.radians(15),
        10.0,
        48,
        (20261001, 20261002, 20261003),
        (1.0, 0.1),
        (0.5, math.radians(2)),
        0.05,
        0.5,
        0.01,
        20.0,
        (
            "bow_height_m",
            "pitch_rad",
            "pitch_rate_rad_s",
            "heave_accel_m_s2",
            "speed_m_s",
            "last_flap",
        ),
        16,
    )
