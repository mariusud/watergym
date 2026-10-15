import math

import torch

from watergym.rare_events import (
    GaussianSea,
    cross_entropy_shift,
    form,
    importance_sampling,
    normal_cdf,
    outcrossing_rate,
    rollout,
)
from watergym.tasks.ride_control import RideControlMoth, sea_state, wand_policy, zero_policy
from watergym.waves import elevation, make_sea


def linear_limit_state(beta: float, dim: int) -> tuple[torch.Tensor, object]:
    direction = torch.randn(dim, generator=torch.Generator().manual_seed(1))
    direction = direction / direction.norm()

    def g(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        margin = beta - z @ direction
        return margin, margin.detach() <= 0

    return direction, g


def test_form_finds_the_design_point_of_a_linear_limit_state() -> None:
    beta, dim = 3.0, 20
    direction, g = linear_limit_state(beta, dim)
    z0 = 0.5 * torch.randn(8, dim, generator=torch.Generator().manual_seed(2))
    result = form(g, z0, iterations=150, lr=0.05)
    assert result.failed.all()
    assert torch.allclose(result.beta, torch.full((8,), beta), rtol=0.02)
    assert torch.allclose(result.z, beta * direction.expand(8, dim), atol=0.1)


def test_importance_sampling_recovers_the_linear_failure_probability() -> None:
    beta, dim = 4.0, 20
    direction, g = linear_limit_state(beta, dim)
    exact = normal_cdf(-beta).item()
    estimate = importance_sampling(lambda z: g(z)[1], beta * direction[None], 4096)
    assert abs(estimate.probability - exact) < 3 * estimate.standard_error
    assert estimate.standard_error < 0.05 * exact


def test_cross_entropy_shift_walks_to_the_failure_region_of_a_linear_limit_state() -> None:
    beta, dim = 4.0, 20
    direction, g = linear_limit_state(beta, dim)
    result = cross_entropy_shift(lambda z: (g(z)[0], z), dim, samples=2000)
    assert result.thresholds[-1] == 0 and all(t > 0 for t in result.thresholds[:-1])
    along = (result.mean.float() @ direction).item()
    assert beta < along < beta + 1.2
    assert (result.mean.float() - along * direction).norm() < 0.5
    estimate = importance_sampling(lambda z: g(z)[1], result.mean[None], 4096)
    assert abs(estimate.probability - normal_cdf(-beta).item()) < 3 * estimate.standard_error


def test_gaussian_sea_has_the_task_spectrum_and_spreading() -> None:
    state = sea_state(0.4, 3.0)
    gaussian = GaussianSea.from_sea_state(state)
    sea = gaussian.sea(torch.randn(20000, gaussian.dim, generator=torch.Generator().manual_seed(0)))
    reference = make_sea(1, state.hs, state.tp, num_components=state.num_components)
    assert torch.allclose((sea.amplitude**2).mean(0), reference.amplitude[0] ** 2, rtol=0.05)
    spread = torch.cos(sea.direction - state.heading_rad).mean()
    assert abs(spread - state.spreading / (state.spreading + 1)) < 0.005
    assert (sea.omega - gaussian.bin_center).abs().max() <= gaussian.bin_width.max() / 2


def test_time_shift_delays_the_sea() -> None:
    gaussian = GaussianSea.from_sea_state(sea_state(0.4, 3.0))
    z = torch.randn(3, gaussian.dim, generator=torch.Generator().manual_seed(0))
    points = torch.tensor([[[1.0, 0.0, 0.0], [-2.0, 0.5, 0.0]]]).expand(3, 2, 3)
    delayed = gaussian.sea(gaussian.time_shift(z, 1.7))
    assert torch.allclose(
        elevation(delayed, points, 5.0), elevation(gaussian.sea(z), points, 5.0 - 1.7), atol=1e-4
    )
    assert torch.allclose(gaussian.time_shift(z, 1.7).norm(dim=-1), z.norm(dim=-1))


def test_time_shift_with_distance_delays_what_a_moving_point_meets() -> None:
    gaussian = GaussianSea.from_sea_state(sea_state(0.4, 3.0))
    z = torch.randn(3, gaussian.dim, generator=torch.Generator().manual_seed(0))
    speed, delay, t = 8.0, 1.7, 5.0
    shifted = gaussian.sea(gaussian.time_shift(z, delay, speed * delay))

    def at(t: float) -> torch.Tensor:
        return torch.tensor([[[speed * t, 0.5, 0.0]]]).expand(3, 1, 3)

    met_before = elevation(gaussian.sea(z), at(t), t)
    assert torch.allclose(elevation(shifted, at(t + delay), t + delay), met_before, atol=1e-4)


def test_outcrossing_rate_of_a_single_wave_is_its_frequency_times_the_tail() -> None:
    # One component, design point all in (u, v): alpha rotates at omega.
    gaussian = GaussianSea.from_sea_state(sea_state(0.4, 3.0))
    z = torch.zeros(1, gaussian.dim)
    z[0, 10] = 3.0
    omega = gaussian.omega(z)[0, 10]
    expected = omega / (2 * math.pi) * math.exp(-4.5)
    assert torch.allclose(outcrossing_rate(gaussian, z).float(), expected[None], rtol=1e-4)


def test_rollout_matches_the_task_and_has_finite_gradients() -> None:
    state = sea_state(0.6, 3.0)
    gaussian = GaussianSea.from_sea_state(state)
    z = torch.randn(4, gaussian.dim, generator=torch.Generator().manual_seed(3))
    steps = 60
    task = RideControlMoth(4, state)
    task.reset(seed=0)
    task.env.sea = gaussian.sea(z)
    obs, _ = task._observe()
    flaps = []
    for _ in range(steps):
        obs, _, terminated, truncated, _ = task.step(wand_policy(obs))
        assert not (terminated | truncated).any()
        flaps.append(obs[:, 5])

    z.requires_grad_(True)
    result = rollout(gaussian, z, wand_policy, steps=steps, checkpoint_steps=20)
    assert not result.crashed.any()
    assert torch.allclose(result.flap, torch.stack(flaps, 1), atol=1e-5)
    assert torch.allclose(result.position_x[:, -1], task.env.eta[:, 0], atol=1e-4)
    assert (result.margin <= result.step_margin.amin(1)).all()
    assert (result.mode_margin >= result.margin[:, None]).all()
    assert (result.mode_margin.amin(-1) - result.margin < 0.005 * math.log(3) + 1e-6).all()
    assert (result.crash_mode == -1).all()
    result.margin.sum().backward()
    assert torch.isfinite(z.grad).all() and (z.grad.norm(dim=-1) > 0).all()


def test_crash_mode_names_the_margin_that_failed() -> None:
    gaussian = GaussianSea.from_sea_state(sea_state(0.4, 3.0))
    z = torch.randn(16, gaussian.dim, generator=torch.Generator().manual_seed(0))
    result = rollout(gaussian, z, zero_policy, steps=300)
    assert result.crashed.any()
    crashed = result.crashed.nonzero()[:, 0]
    at_crash = result.step_margin[crashed, result.crash_step[crashed]]
    assert (at_crash < 0).all()
    assert (result.crash_mode[crashed] >= 0).all() and (
        result.crash_mode[~result.crashed] == -1
    ).all()
