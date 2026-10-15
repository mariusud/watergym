import math

import pytest
import torch

from watergym import SeaState, WaterEnv
from watergym.current import Current, body_current
from watergym.foils import Foil, foil_loads
from watergym.rigid_body import Pose, RigidBody, acceleration, kinematics, rk4_step
from watergym.vessels import box_barge, moth, otter
from watergym.vessels.box_barge_vessel import MASS_KG
from watergym.vessels.moth_vessel import wand_action
from watergym.waves import regular_wave


def test_floating_box_drifts_toward_the_current_speed() -> None:
    """Surge with quadratic drag d only: M du/dt = d (U - u)^2, so from rest
    U - u = U / (1 + d U t / M), with M the mass plus surge added mass."""
    env = WaterEnv(box_barge(), 2, SeaState(hs=0.0), dt=0.1)
    env.reset()
    speeds = torch.tensor([0.5, 1.0])
    env.sea.current = Current.uniform(2, speeds)
    for _ in range(200):
        env.step(None)
    drag = env.vessel.body.quadratic_damping[0]
    lag = speeds / (1 + drag * speeds * env.t / (1.05 * MASS_KG))
    world_velocity = kinematics(env.eta, env.nu, Pose.from_eta(env.eta))
    assert torch.allclose(world_velocity[:, 0], speeds - lag, rtol=1e-3)
    assert world_velocity[:, 1].abs().max() < 1e-6


def test_zero_current_is_bit_identical_to_no_current() -> None:
    def run(current: Current | None) -> torch.Tensor:
        env = WaterEnv(moth(), 4, SeaState(hs=0.4, tp=3.0, heading_rad=math.pi))
        env.reset(seed=3)
        env.sea.current = current
        for _ in range(50):
            env.step(wand_action(env.sea, env.t, env.eta))
        return torch.cat((env.eta, env.nu), -1)

    assert torch.equal(run(None), run(Current.uniform(4, 0.0, direction=1.0)))


def test_station_keeping_thrust_equals_the_current_drag() -> None:
    """Bow into a 0.5 m/s current, the thrusters hold the steady drag d_u U and the hull
    stays put. Without thrust it drifts at the current speed."""
    vessel = otter()
    speed, drag = 0.5, vessel.body.linear_damping[0].item() * 0.5
    command = drag / sum(thruster.max_force_n for thruster in vessel.thrusters)

    def drift(command: float) -> float:
        env = WaterEnv(vessel, 1, SeaState(hs=0.0))
        env.reset()
        env.sea.current = Current.uniform(1, speed, direction=math.pi)
        for _ in range(250):
            env.step(torch.full((1, 2), command))
        return kinematics(env.eta, env.nu, Pose.from_eta(env.eta))[0, 0].item()

    assert drift(command) == pytest.approx(0.0, abs=1e-3 * speed)
    assert drift(0.0) == pytest.approx(-speed, rel=2e-2)


def test_foils_feel_the_current_like_their_own_speed() -> None:
    """A foil held still in a current toward -x sees the flow of one moving forward through
    still water. Half way down a profile from 1 at the surface to 0 at 1 m, a 10 m/s surface
    current gives the same flow as 5 m/s."""
    foil = Foil("main", (0.0, 0.0, 0.5), span_m=1.0, chord_m=0.1, incidence=0.05)
    calm, pose = regular_wave(1, 0.0, 5.0), Pose.from_eta(torch.zeros(1, 6))
    flap, dry = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    moving = foil_loads(foil, calm, 0.0, pose, torch.tensor([[5.0, 0, 0, 0, 0, 0]]), flap, dry)

    calm.current = Current.uniform(1, 5.0, direction=math.pi)
    held = foil_loads(foil, calm, 0.0, pose, torch.zeros(1, 6), flap, dry)
    surface = Current.uniform(1, 10.0, direction=math.pi)
    calm.current = Current(
        surface.speed, surface.direction, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    )
    sheared = foil_loads(foil, calm, 0.0, pose, torch.zeros(1, 6), flap, dry)
    assert moving.force.norm() > 100
    assert torch.allclose(held.force, moving.force, rtol=1e-5)
    assert torch.allclose(sheared.force, moving.force, rtol=1e-5)


def test_a_spinning_body_carried_by_the_current_stays_with_the_water() -> None:
    """nu_r = 0 solves the equations for a body with unequal added masses drifting with the
    current while it yaws, if the current's body-frame rate M_A nu_c_dot is included."""
    body = RigidBody.from_diagonals(
        100.0, [10.0] * 3, [20.0, 80.0, 80.0, 1, 1, 1], [0.0] * 6, [0.0] * 6
    )
    current = Current.uniform(1, 1.0, direction=0.3)

    def derivative(t, eta, nu):
        pose = Pose.from_eta(eta)
        return kinematics(eta, nu, pose), acceleration(
            body, nu, torch.zeros(1, 6), body_current(current, pose, nu)
        )

    eta = torch.zeros(1, 6)
    nu = torch.tensor([[math.cos(0.3), math.sin(0.3), 0.0, 0.0, 0.0, 0.8]])
    for _ in range(500):
        eta, nu = rk4_step(derivative, torch.zeros(1), eta, nu, 0.01)
    world_velocity = kinematics(eta, nu, Pose.from_eta(eta))[0, :3]
    expected = torch.tensor([math.cos(0.3), math.sin(0.3), 0.0])
    assert torch.allclose(world_velocity, expected, atol=1e-4)


def test_current_rate_matches_a_finite_difference_through_a_depth_profile() -> None:
    current = Current(
        torch.tensor([0.8]).double(),
        torch.tensor([0.4]).double(),
        torch.tensor([0.0, 2.0]).double(),
        torch.tensor([1.0, 0.2]).double(),
    )
    eta = torch.tensor([[0.0, 0.0, 1.0, 0.1, -0.2, 0.5]], dtype=torch.float64)
    nu = torch.tensor([[0.3, -0.1, 0.4, 0.2, -0.3, 0.5]], dtype=torch.float64)
    eta_dot = kinematics(eta, nu, Pose.from_eta(eta))
    h = 1e-5
    ahead, _ = body_current(current, Pose.from_eta(eta + h * eta_dot), nu)
    behind, _ = body_current(current, Pose.from_eta(eta - h * eta_dot), nu)
    _, rate = body_current(current, Pose.from_eta(eta), nu)
    assert torch.allclose(rate, (ahead - behind) / (2 * h), atol=1e-8)
