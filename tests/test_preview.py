import math

import pytest
import torch

from watergym.tasks import TASKS, make
from watergym.tasks.ride_control import TASK_ID, sea_state, wand_policy
from watergym.tasks.ride_control_preview import (
    BEAM_HALF_ANGLE,
    DROPOUT_END_PROBABILITY,
    DROPOUT_START_PROBABILITY,
    PREVIEW_COUNTS,
    PREVIEW_LENGTHS_M,
    RideControlPreviewMoth,
    footprint_points,
    preview_distances,
    slant_ranges,
    task_id,
)
from watergym.vessels.moth_vessel import WAND_PIVOT
from watergym.waves import regular_wave


def test_readings_are_spread_from_half_a_metre_to_the_preview_length() -> None:
    assert preview_distances(1, 4.0).tolist() == [4.0]
    assert preview_distances(4, 2.0).tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0])


def test_footprint_is_centred_ahead_of_the_bow_sensor_on_the_body_x_axis() -> None:
    distances = torch.tensor([0.5, 8.0])
    points = footprint_points(distances)
    centres = points[:, 0]
    expected = torch.tensor([[WAND_PIVOT[0] + d, WAND_PIVOT[1], WAND_PIVOT[2]] for d in (0.5, 8.0)])
    assert torch.allclose(centres, expected)
    radius = (points[:, 1:] - centres[:, None]).norm(dim=-1)
    expected = slant_ranges(distances) * math.tan(BEAM_HALF_ANGLE)
    assert radius == pytest.approx(expected[:, None].expand(2, 4), rel=1e-5)


def test_a_reading_measures_the_surface_where_the_pitched_body_x_axis_lands_in_ned() -> None:
    """A long-crested wave much longer than the footprint, the boat pitched bow up 5 deg:
    the reading is the sensor's NED height minus the analytic elevation at the point
    L along the body x axis ahead of the sensor."""
    length_m, theta = 8.0, math.radians(5)
    task = RideControlPreviewMoth(1, k=1, length_m=length_m)
    amplitude, period = 0.3, 12.0
    task.env.sea = regular_wave(1, amplitude, period)
    task.env.eta[0] = torch.tensor([2.0, 0.0, -0.85, 0.0, theta, 0.0])
    task.env.t[0] = 1.3

    px, pz = WAND_PIVOT[0] + length_m, WAND_PIVOT[2]
    x_hit = 2.0 + px * math.cos(theta) + pz * math.sin(theta)
    sensor_z = -0.85 - WAND_PIVOT[0] * math.sin(theta) + WAND_PIVOT[2] * math.cos(theta)
    omega = 2 * math.pi / period
    k = omega**2 / 9.81
    surface = amplitude * math.cos(k * x_hit - omega * 1.3)
    assert task.true_readings()[0, 0].item() == pytest.approx(-sensor_z - surface, abs=2e-4)


def test_in_calm_water_every_reading_is_the_bow_sensor_height() -> None:
    task = RideControlPreviewMoth(2, sea_state(0.0, 3.0), k=4, length_m=4.0)
    task.reset(seed=0)
    readings = task.true_readings()
    assert readings == pytest.approx(readings[:, :1].expand(2, 4), abs=1e-5)


def test_range_noise_has_the_documented_standard_deviation() -> None:
    task = RideControlPreviewMoth(4096, k=2, length_m=8.0)
    noise = task._noisy(torch.zeros(4096, 2))
    assert noise.mean(0).abs().max() < 0.003
    assert noise.std(0) == pytest.approx(task.noise_std, rel=0.05)


def test_dropouts_come_in_bursts_and_hold_the_last_good_reading() -> None:
    task = RideControlPreviewMoth(256, sea_state(0.4, 3.0), k=2, length_m=2.0)
    task.reset(seed=1)
    dropped, held = [], []
    for _ in range(400):
        before = task.preview.clone()
        task._read_preview()
        dropped.append(task.dropped.clone())
        held.append(torch.equal(task.preview[task.dropped], before[task.dropped]))
    dropped = torch.stack(dropped[50:])
    stationary = DROPOUT_START_PROBABILITY / (DROPOUT_START_PROBABILITY + DROPOUT_END_PROBABILITY)
    assert dropped.float().mean().item() == pytest.approx(stationary, rel=0.1)
    bursts = (dropped[1:] & ~dropped[:-1]).sum()
    mean_burst = dropped[1:].sum() / bursts
    assert mean_burst.item() == pytest.approx(1 / DROPOUT_END_PROBABILITY, rel=0.1)
    assert all(held)


def test_without_readings_the_task_is_v1() -> None:
    v1 = make(TASK_ID, 3, sea_state(0.4, 3.0))
    preview = make(task_id(0, 1.0), 3, sea_state(0.4, 3.0))
    obs_v1, _ = v1.reset(seed=7)
    obs, _ = preview.reset(seed=7)
    for _ in range(20):
        obs_v1, reward_v1, *_ = v1.step(wand_policy(obs_v1))
        obs, reward, *_ = preview.step(wand_policy(obs))
    assert torch.equal(obs, obs_v1) and torch.equal(reward, reward_v1)


def test_the_readings_extend_both_observations_without_changing_the_seas() -> None:
    v1 = make(TASK_ID, 2, sea_state(0.4, 3.0))
    task = make(task_id(4, 8.0), 2, sea_state(0.4, 3.0))
    assert (task.observation_size, task.privileged_observation_size) == (10, 20)
    obs_v1, _ = v1.reset(seed=3)
    obs, info = task.reset(seed=3)
    assert obs.shape == (2, 10) and info["privileged_obs"].shape == (2, 20)
    obs_v1, *_ = v1.step(wand_policy(obs_v1))
    obs, _, _, _, info = task.step(wand_policy(obs))
    assert torch.equal(obs[:, :6], obs_v1)
    assert torch.equal(info["privileged_obs"][:, 16:], task.true_preview)


def test_the_grid_is_registered() -> None:
    for k in PREVIEW_COUNTS:
        for length_m in PREVIEW_LENGTHS_M:
            assert task_id(k, length_m) in TASKS
    assert task_id(8, 2.0) == "RideControlPreview-Moth-v1-k8-L2"
