import math

import pytest
import torch

from watergym.rigid_body import (
    Pose,
    RigidBody,
    acceleration,
    coriolis_force,
    kinematics,
    rk4_step,
)


def test_free_fall_matches_analytic() -> None:
    body = RigidBody.from_diagonals(2.0, [1.0] * 3, [0.0] * 6, [0.0] * 6, [0.0] * 6)
    weight = torch.tensor([[0.0, 0.0, 2.0 * 9.81, 0.0, 0.0, 0.0]])

    def derivative(t, eta, nu):
        return kinematics(eta, nu, Pose.from_eta(eta)), acceleration(body, nu, weight)

    eta, nu, t = torch.zeros(1, 6), torch.zeros(1, 6), torch.zeros(1)
    for _ in range(100):
        eta, nu = rk4_step(derivative, t, eta, nu, 0.01)
    assert eta[0, 2].item() == pytest.approx(0.5 * 9.81, rel=1e-5)
    assert nu[0, 2].item() == pytest.approx(9.81, rel=1e-5)


def damped_oscillator_error(dt: float) -> float:
    omega_n, zeta = 2 * math.pi, 0.1

    def derivative(t, x, v):
        return v, -2 * zeta * omega_n * v - omega_n**2 * x

    x, v = torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)
    t = torch.zeros(1, dtype=torch.float64)
    for _ in range(round(5.0 / dt)):
        x, v = rk4_step(derivative, t, x, v, dt)
    omega_d = omega_n * math.sqrt(1 - zeta**2)
    exact = math.exp(-zeta * omega_n * 5) * (
        math.cos(omega_d * 5) + zeta * omega_n / omega_d * math.sin(omega_d * 5)
    )
    return abs(x.item() - exact)


def test_rk4_is_fourth_order() -> None:
    assert damped_oscillator_error(0.01) < 1e-6
    assert 12 < damped_oscillator_error(0.01) / damped_oscillator_error(0.005) < 20


def test_coriolis_does_no_work() -> None:
    mass_matrix = torch.diag(torch.tensor([80.0, 160.0, 130.0, 25.0, 45.0, 40.0]))
    nu = torch.randn(32, 6)
    power = (coriolis_force(mass_matrix, nu) * nu).sum(-1)
    assert power.abs().max().item() < 1e-3


def test_rotation_maps_body_forward_to_heading() -> None:
    eta = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, math.pi / 2]])
    nose = Pose.from_eta(eta).to_world(torch.tensor([[1.0, 0.0, 0.0]]))
    assert nose[0, 0].tolist() == pytest.approx([1.0, 3.0, 0.0], abs=1e-6)
