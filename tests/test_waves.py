import math

import pytest
import torch

from watergym.waves import (
    elevation,
    jonswap,
    make_sea,
    orbital_velocity,
    regular_wave,
    significant_wave_height,
)


def test_jonswap_peaks_at_peak_frequency() -> None:
    omega = torch.linspace(0.2, 3.0, 20001)
    spectrum = jonswap(omega, torch.tensor(2.0), torch.tensor(8.0))
    assert omega[spectrum.argmax()].item() == pytest.approx(2 * math.pi / 8.0, rel=0.01)


@pytest.mark.parametrize("hs", [0.3, 2.0, 5.0])
@pytest.mark.parametrize("tp", [5.0, 8.0, 12.0])
def test_components_carry_the_requested_hs(hs: float, tp: float) -> None:
    sea = make_sea(4, hs, tp, num_components=64)
    assert significant_wave_height(sea) == pytest.approx(torch.full((4,), hs), rel=0.01)


def test_surface_record_has_the_requested_hs() -> None:
    sea = make_sea(8, 2.0, 8.0, num_components=128, spreading=10.0)
    times = torch.arange(0, 1800, 0.5)
    origin = torch.zeros(8, 1, 3)
    record = torch.stack([elevation(sea, origin, t)[:, 0] for t in times], dim=-1)
    assert 4 * record.std(dim=-1).mean().item() == pytest.approx(2.0, rel=0.03)


def test_orbital_velocity_matches_airy_theory() -> None:
    amplitude, period, depth = 0.5, 6.0, 3.0
    sea = regular_wave(1, amplitude, period)
    omega = 2 * math.pi / period
    k = omega**2 / 9.81
    crest_below = torch.tensor([[[0.0, 0.0, depth]]])
    velocity = orbital_velocity(sea, crest_below, 0.0)[0, 0]
    assert velocity[0].item() == pytest.approx(amplitude * omega * math.exp(-k * depth), rel=1e-5)
    assert velocity[2].item() == pytest.approx(0.0, abs=1e-6)


def test_surface_rises_at_the_vertical_orbital_velocity() -> None:
    sea = make_sea(2, 1.0, 6.0, num_components=16)
    point = torch.tensor([[[3.0, -2.0, 0.0]]]).expand(2, 1, 3)
    t, dt = 4.0, 1e-3
    rise_rate = (elevation(sea, point, t + dt) - elevation(sea, point, t - dt)) / (2 * dt)
    upward_velocity = -orbital_velocity(sea, point, t)[..., 2]
    assert rise_rate == pytest.approx(upward_velocity, rel=1e-3, abs=1e-4)


def test_subset_keeps_the_selected_envs_for_slices_lists_and_tensors() -> None:
    sea = make_sea(5, 1.0, 6.0, heading_rad=math.pi)
    for index in (slice(1, 3), [1, 2], torch.tensor([1, 2])):
        part = sea.subset(index)
        assert part.num_envs == 2
        assert torch.equal(part.phase, sea.phase[1:3])
        assert torch.equal(part.amplitude, sea.amplitude[1:3])


def test_heading_rad_sets_the_wave_direction() -> None:
    assert (make_sea(2, 1.0, 6.0, heading_rad=math.pi).direction == math.pi).all()
    assert (regular_wave(2, 1.0, 6.0, heading_rad=math.pi / 2).direction == math.pi / 2).all()
