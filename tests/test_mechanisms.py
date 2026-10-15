import pytest
import torch

from watergym.mechanisms import CAM_KNOTS, MechanicalWand, cam, stock_linkage
from watergym.tasks import make
from watergym.tasks.ride_control import TASK_ID, sea_state, wand_policy
from watergym.tasks.ride_control_variants import RideControlBowOnlyMoth
from watergym.vessels.moth_vessel import WAND_GEARING, WAND_NEUTRAL_ANGLE


def test_stock_cam_is_the_linear_gearing() -> None:
    linkage = stock_linkage(3)
    angles = torch.tensor([0.3, WAND_NEUTRAL_ANGLE, 1.2])
    expected = WAND_GEARING * (WAND_NEUTRAL_ANGLE - angles)
    assert torch.allclose(cam(linkage.cam_flap, angles), expected, atol=1e-6)


def test_cam_is_monotone_between_uneven_knots() -> None:
    knots = torch.tensor([[0.1, 0.09, 0.0, -0.01, -0.1]])
    angles = torch.linspace(CAM_KNOTS[0].item(), CAM_KNOTS[-1].item(), 400)
    flaps = cam(knots.expand(400, -1), angles)
    assert (flaps.diff() <= 1e-7).all()
    assert torch.allclose(cam(knots.expand(5, -1), CAM_KNOTS), knots[0], atol=1e-6)


def fly(controller, hs: float, tp: float, steps: int, mechanical: bool):
    task = make(TASK_ID, 4, sea_state(hs, tp))
    obs, _ = task.reset(seed=3)
    heights, flaps, contact = [], [], []
    for _ in range(steps):
        flap = controller(task) if mechanical else controller(obs)
        obs, _, _, _, info = task.step(flap)
        heights.append(info["signals"]["height_error_m"])
        flaps.append(info["signals"]["flap_deg"])
        if mechanical:
            contact.append(controller.in_contact.clone())
    return torch.stack(heights), torch.stack(flaps), contact


def test_dynamic_wand_trims_like_the_static_wand_in_calm_water() -> None:
    static_height, static_flap, _ = fly(wand_policy, 0.0, 3.0, 300, mechanical=False)
    wand = MechanicalWand(stock_linkage(4))
    height, flap, contact = fly(wand, 0.0, 3.0, 300, mechanical=True)
    assert abs(height[-50:].mean() - static_height[-50:].mean()) < 0.01
    assert abs(flap[-50:].mean() - static_flap[-50:].mean()) < 0.2
    assert contact[-1].all()


def test_dynamic_wand_skips_in_short_steep_chop() -> None:
    wand = MechanicalWand(stock_linkage(4))
    _, _, contact = fly(wand, 0.3, 1.5, 200, mechanical=True)
    airborne = 1 - torch.stack(contact).float().mean()
    assert airborne > 0.05


def test_bow_only_task_sees_bow_height_and_last_flap() -> None:
    task = make(RideControlBowOnlyMoth.task_id, 2, sea_state(0.2, 3.0))
    full = make(TASK_ID, 2, sea_state(0.2, 3.0))
    obs, info = task.reset(seed=0)
    full_obs, full_info = full.reset(seed=0)
    for _ in range(3):
        obs, reward, _, _, info = task.step(torch.full((2, 1), 0.3))
        full_obs, full_reward, _, _, full_info = full.step(torch.full((2, 1), 0.3))
    assert obs.shape == (2, 2)
    assert torch.equal(obs, full_obs[:, [0, 5]])
    assert torch.equal(reward, full_reward)
    assert torch.equal(info["privileged_obs"], full_info["privileged_obs"])
    assert obs[0, 1].item() == pytest.approx(0.3)
