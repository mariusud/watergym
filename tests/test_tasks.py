import math

import pytest
import torch

from watergym.tasks import TASKS, make, ride_control
from watergym.tasks.metrics import analog_gain, per_minute, rms, wk_analog_sections, wk_filter
from watergym.tasks.ride_control import TASK_ID, sea_state, wand_policy, zero_policy
from watergym.vessels import moth, moth_vessel
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
    10.0: 988,
    12.5: 902,
    16.0: 768,
    20.0: 636,
    25.0: 513,
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


@pytest.mark.parametrize("frequency", [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0])
def test_wk_filter_weights_a_sinusoid_like_the_iso_table_up_to_nyquist(frequency: float) -> None:
    """At 50 Hz sampling, up to the 25 Hz Nyquist frequency. A cosine, since a sine sampled
    at Nyquist is zero. The first and last 10 s hold the start and stop transients."""
    t = torch.arange(0, 60, 0.02)[:, None]
    signal = torch.cos(2 * math.pi * frequency * t)
    weighted = wk_filter(signal, sample_rate=50.0)
    gain = rms(weighted[500:-500]) / rms(signal[500:-500])
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


def test_signals_survive_the_reset_of_a_crashed_env() -> None:
    """A reset clears env.ventilated in place; the step's signals must keep their values."""
    task = make(TASK_ID, 2, sea_state(0.0, 3.0))
    task.reset(seed=0)
    _, _, _, _, info = task.step(torch.zeros(2, 1))
    signals = {k: v.clone() for k, v in info["signals"].items()}
    task.env.ventilated[:] = True
    task.env._reset_envs(torch.arange(2))
    for name, value in signals.items():
        assert torch.equal(info["signals"][name], value), name


def test_evaluation_is_deterministic_for_a_seed() -> None:
    first, second = (run(wand_policy, 50, hs=0.4, seed=7) for _ in range(2))
    assert torch.equal(first[-1]["height_error_m"], second[-1]["height_error_m"])


def test_heave_accel_is_the_derivative_of_the_heave_rate() -> None:
    """With the flap held, the observed acceleration matches a central difference of the
    world vertical velocity: gravity is removed and the frame is right."""
    task = make(TASK_ID, 2, sea_state(0.3, 3.0))
    obs, _ = task.reset(seed=3)
    rates, accels = [], []
    for _ in range(60):
        obs, *_ = task.step(torch.full((2, 1), 0.2))
        rates.append(task.heave_rate.clone())
        accels.append(obs[:, 3])
    rates, accels = torch.stack(rates), torch.stack(accels)
    central = (rates[2:] - rates[:-2]) / (2 * ride_control.DT)
    assert torch.allclose(accels[1:-1], central, atol=0.1)
    assert accels.abs().max() > 2.0


def test_v1_definition_is_frozen() -> None:
    """Changing any of these changes RideControl-Moth-v1. Make RideControl-Moth-v2 instead."""
    assert TASKS["RideControl-Moth-v1"] is ride_control.RideControlMoth
    main_foil = moth().foils[0]
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
        # The Moth the task flies.
        main_foil.position,
        round(math.degrees(main_foil.incidence), 6),
        moth().thrusters[0].position,
        (moth_vessel.WAND_GEARING, moth_vessel.WAND_NEUTRAL_ANGLE, moth_vessel.WAND_PIVOT),
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
        0.005,
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
        (0.45, 0.0, 1.25),
        3.2,
        (-0.4, 0.0, -2.0),
        (0.133, math.radians(45), (1.5, 0.0, 0.1)),
    )
