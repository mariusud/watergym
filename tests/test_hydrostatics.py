import dataclasses
import math

import pytest
import torch

from watergym.env import SeaState, WaterEnv
from watergym.hydrostatics import hydrostatic_force
from watergym.rigid_body import Pose
from watergym.vessel import gravity_force
from watergym.vessels.box_barge_vessel import BEAM_M, LENGTH_M, MASS_KG, box_barge
from watergym.waves import regular_wave

CALM = regular_wave(1, amplitude=0.0, period=5.0)
RHO_G = 1025.0 * 9.81


def net_force(eta: list[float]) -> torch.Tensor:
    barge = box_barge()
    pose = Pose.from_eta(torch.tensor([eta]))
    return (gravity_force(barge.body.mass, pose) + hydrostatic_force(barge.hull, CALM, 0.0, pose))[
        0
    ]


def test_barge_floats_at_analytic_draft() -> None:
    at_rest = net_force(list(box_barge().initial_eta))
    assert at_rest.abs().max().item() < 1e-4 * MASS_KG * 9.81


def test_heave_stiffness_is_rho_g_waterplane() -> None:
    sunk = list(box_barge().initial_eta)
    sunk[2] += 0.05
    assert -net_force(sunk)[2].item() / 0.05 == pytest.approx(RHO_G * LENGTH_M * BEAM_M, rel=1e-3)


def test_roll_restoring_moment_uses_metacentric_height() -> None:
    heeled = list(box_barge().initial_eta)
    heeled[3] = math.radians(1)
    gm = 0.8333
    expected = MASS_KG * 9.81 * gm * math.sin(math.radians(1))
    assert -net_force(heeled)[3].item() == pytest.approx(expected, rel=0.02)


def test_heave_free_decay_period() -> None:
    barge = box_barge(heave_damping_ratio=0.0)
    env = WaterEnv(barge, 1, SeaState(hs=0.0), dt=0.02, substeps=1)
    env.reset()
    env.eta[:, 2] += 0.1
    action = torch.zeros(1, 0)
    heave, times = [], []
    for _ in range(500):
        env.step(action)
        heave.append(env.eta[0, 2].item() - barge.initial_eta[2])
        times.append(env.t[0].item())
    upward_crossings = [
        times[i - 1] + (times[i] - times[i - 1]) * heave[i - 1] / (heave[i - 1] - heave[i])
        for i in range(1, len(heave))
        if heave[i - 1] < 0 <= heave[i]
    ]
    period = (upward_crossings[-1] - upward_crossings[0]) / (len(upward_crossings) - 1)
    added_mass = barge.body.added_mass[2, 2].item()
    expected = 2 * math.pi * math.sqrt((MASS_KG + added_mass) / (RHO_G * LENGTH_M * BEAM_M))
    assert period == pytest.approx(expected, rel=0.01)


def test_long_waves_lift_the_barge_with_the_surface() -> None:
    barge = dataclasses.replace(box_barge(), free_dofs=(False, False, True, False, False, False))
    env = WaterEnv(barge, 1, SeaState(hs=0.0), dt=0.05, substeps=2, episode_length_s=100.0)
    env.reset()
    env.sea = regular_wave(1, amplitude=0.05, period=30.0)
    heave = []
    for _ in range(1200):
        env.step(torch.zeros(1, 0))
        heave.append(env.eta[0, 2].item() - barge.initial_eta[2])
    steady = torch.tensor(heave[600:])
    assert (steady.max() - steady.min()).item() / 2 == pytest.approx(0.05, rel=0.03)
