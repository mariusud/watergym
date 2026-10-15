import math

import pytest
import torch

from watergym.env import SeaState, WaterEnv
from watergym.rigid_body import Pose
from watergym.vessels import box_barge, moth
from watergym.wind import (
    GUST_BAND_HZ,
    WindState,
    froya_spectrum,
    make_wind,
    wind_force,
    windage_from_mesh,
)


@pytest.mark.parametrize("speed", [5.0, 15.0, 25.0])
def test_gust_variance_matches_the_froya_spectrum_over_the_band(speed: float) -> None:
    wind = make_wind(3, speed)
    low, high = GUST_BAND_HZ
    f = torch.logspace(math.log10(low), math.log10(high), 100_001, dtype=torch.float64)
    target = torch.trapezoid(froya_spectrum(f, torch.tensor(speed, dtype=torch.float64)), f)
    realized = (wind.amplitude**2 / 2).sum(-1)
    assert realized == pytest.approx(torch.full((3,), target.item()), rel=0.01)


def test_floating_box_drifts_downwind() -> None:
    east = WindState(speed=15.0, direction_rad=math.pi / 2, gusts=False)
    calm = WaterEnv(box_barge(), 2, dt=0.05)
    windy = WaterEnv(box_barge(), 2, dt=0.05, wind_state=east)
    for env in (calm, windy):
        env.reset(seed=0)
        for _ in range(100):
            env.step(None)
    assert (windy.eta[:, 1] > 0.05).all()
    assert windy.eta[:, 0].abs().max() < 0.05 * windy.eta[:, 1].min()
    assert calm.eta[:, 1].abs().max() < 1e-4


def test_wind_force_scales_with_speed_squared() -> None:
    barge = box_barge()
    windage = windage_from_mesh(barge.mesh, barge.initial_eta)
    pose = Pose.from_eta(torch.tensor([barge.initial_eta]))
    at_rest = torch.zeros(1, 6)
    forces = [
        wind_force(make_wind(1, speed, 0.7, gusts=False), windage, torch.zeros(1), pose, at_rest)
        for speed in (5.0, 10.0)
    ]
    assert (forces[0][0, [0, 1, 3, 4, 5]].abs() > 0).all()
    assert forces[1] == pytest.approx(4 * forces[0], rel=1e-5)


def test_no_wind_and_zero_wind_are_bit_identical() -> None:
    seas = SeaState(hs=0.3, tp=3.0)
    envs = [
        WaterEnv(moth(), 2, seas),
        WaterEnv(moth(), 2, seas, wind_state=WindState(speed=0.0, direction_rad=1.0)),
    ]
    for env in envs:
        env.reset(seed=4)
        for _ in range(25):
            env.step(torch.full((2, 2), 0.5))
    assert torch.equal(envs[0].eta, envs[1].eta)
    assert torch.equal(envs[0].nu, envs[1].nu)
